from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pdb
from typing import Dict, Tuple

warnings.filterwarnings('ignore')


from torchdiffeq import odeint # 直接导入，如果未安装则会报错
# ========================= 度量与ODE模块 =========================
class DiagMetric(nn.Module):
    """位置依赖的黎曼度量（对角SPD矩阵）: M(h) = diag(softplus(m(h))) + eps"""
    def __init__(self, d: int, hidden: int = 512, eps: float = 1e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, d)
        )
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, d]
        m = F.softplus(self.net(h)) + self.eps  # 确保结果为正
        return m  # [B, d], M的对角线元素

class ODEVF(nn.Module):
    """黑盒向量场 dh/dt = f_theta(h, t)"""
    def __init__(self, d: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, d)
        )

    def forward(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # 将时间t作为一个特征拼接到h上
        t_feat = torch.ones(h.size(0), 1, device=h.device) * t
        return self.net(torch.cat([h, t_feat], dim=-1))

class MetricGradientFlow(nn.Module):
    """
    几何向量场 (度量梯度流):
        dh/dt = - M(h)^{-1} * ∇_h U(h; batch, y)
    U 是一个类似 InfoNCE 的势函数，它将同类样本拉近，将不同类样本推远。
    """
    def __init__(self, d: int, metric_net: DiagMetric, temp: float = 0.07):
        super().__init__()
        self.metric_net = metric_net
        self.temp = temp

    def potential_and_grad(self, h: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用 autograd 计算势函数 U 和其欧氏梯度 ∇_h U"""
        h = F.normalize(h, dim=-1)
        B = h.size(0)
        sim = (h @ h.t()) / self.temp  # [B, B]
        eye = torch.eye(B, device=h.device, dtype=torch.bool)
        sim = sim.masked_fill(eye, -6e4)  # 屏蔽对角线元素，该值对 float16 友好

        same = (y.unsqueeze(1) == y.unsqueeze(0)).float()
        # 多正例 InfoNCE: U = -(pos - logsumexp(all))
        logZ = torch.logsumexp(sim, dim=1)
        pos = torch.logsumexp(sim + (same + 1e-6).log(), dim=1)
        U = -(pos - logZ).mean()

        grad_h = torch.autograd.grad(U, h, retain_graph=True, create_graph=True)[0]
        return U, grad_h

    def forward(self, t: torch.Tensor, h: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        # 如果没有提供标签 (例如在推理时)，向量场为零，h 不会改变
        if y is None:
            return torch.zeros_like(h)
        m_diag = self.metric_net(h)         # [B, d]
        U, grad_h = self.potential_and_grad(h.requires_grad_(), y)
        inv_m = 1.0 / m_diag                # 对角矩阵的逆
        return - inv_m * grad_h             # dh/dt


class ODEBlock(nn.Module):
    """一个包装器，通过 torchdiffeq 将向量场从 t=0 积分到 t=T"""
    def __init__(self, vf: nn.Module, T: float = 1.0, solver: str = 'dopri5', rtol: float = 1e-3, atol: float = 1e-4):
        super().__init__()
        self.vf = vf
        self.T = T
        self.solver = solver
        self.rtol = rtol
        self.atol = atol

    def forward(self, h0: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        # 为 MetricGradientFlow 包装一下，使其符合 odeint 的 (t, h) 输入格式
        if isinstance(self.vf, MetricGradientFlow):
            func = lambda t, h: self.vf(t, h, y)
        else:
            func = self.vf
        
        t_span = torch.tensor([0., self.T], device=h0.device)
        # 调用 odeint 求解器，只取最终时刻 T 的结果
        hT = odeint(func, h0, t_span, method=self.solver, rtol=self.rtol, atol=self.atol)[-1]
        return hT


# ========================= 模型包装器 =========================
class ResNetNeuralODEMetric(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = 512,
                 pretrained: bool = True,
                 use_metric_flow: bool = True,
                 use_ode: bool = True,
                 T: float = 1.0,
                 solver: str = 'dopri5',
                 backbone_name: str = "resnet18",   # ✅ 新增参数，用于选择主干网络
                 ):
        super().__init__()
        self.use_ode = use_ode

        # -------- 主干网络 --------
        # 自动从 torchvision.models 加载主干网络
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.get_model(backbone_name, weights=weights)

        # 动态替换最后的分类头，以提取指定维度的特征
        if hasattr(self.backbone, "fc"):  # ResNet 系列
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, feat_dim)
        elif hasattr(self.backbone, "classifier"): # MobileNet, EfficientNet 等
            if isinstance(self.backbone.classifier, nn.Sequential):
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier[-1] = nn.Linear(in_features, feat_dim)
            else:
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, feat_dim)
        elif hasattr(self.backbone, "head"):  # Swin Transformer (来自timm库) 系列
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, feat_dim)
        elif hasattr(self.backbone, "heads"): # Vision Transformer (ViT) 系列
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(in_features, feat_dim)
        else:
            raise ValueError(f"不支持的主干网络: {backbone_name}")

        # -------- 度量与ODE模块 --------
        self.metric = DiagMetric(d=feat_dim)
        if use_metric_flow:
            vf = MetricGradientFlow(d=feat_dim, metric_net=self.metric)
        else:
            vf = ODEVF(d=feat_dim) # 使用一个普通的黑盒向量场
        self.ode = ODEBlock(vf, T=T, solver=solver)
            
        # -------- 分类头 --------
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None):
        h0 = self.backbone(x)     # 初始特征 [B, d]
        hT = self.ode(h0, y=y)    # 通过ODE演化后的特征 [B, d]
        
        # 根据 use_ode 参数决定使用哪个特征进行分类
        if self.use_ode:
            logits = self.head(hT)
        else:
            logits = self.head(h0)
        return h0, hT, logits


# ========================= 损失函数 =========================

def metric_infoNCE_g(hT: torch.Tensor, y: torch.Tensor, metric_net: DiagMetric, temp: float = 0.07) -> torch.Tensor:
    """
    使用 g-距离计算的 InfoNCE 损失:
        d_g^2(h_i,h_j) = (h_i - h_j)^T M((h_i+h_j)/2) (h_i - h_j)
    为提高效率，这里使用对角矩阵 M。
    """
    B, d = hT.shape
    # 计算所有点对的中点 (h_i+h_j)/2 处的度量 M
    with torch.no_grad():
        h_mid = (hT.unsqueeze(1) + hT.unsqueeze(0)) / 2.0       # [B,B,d]
        m_mid = metric_net(h_mid.reshape(-1, d)).reshape(B, B, d)

    diff = hT.unsqueeze(1) - hT.unsqueeze(0)                    # [B,B,d]
    dist2 = (diff * m_mid * diff).sum(-1)                       # [B,B], g-距离的平方
    sim = -dist2 / temp                                         # 将距离转化为相似度

    eye = torch.eye(B, device=hT.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9) # 屏蔽对角线
    same = (y.unsqueeze(1) == y.unsqueeze(0)).float()

    logZ = torch.logsumexp(sim, dim=1)
    pos = torch.logsumexp(sim + (same + 1e-6).log(), dim=1)
    loss = -(pos - logZ).mean()
    return loss



class Exp_MetricLearning(Exp_Basic):
    def __init__(self, args):
        super(Exp_MetricLearning, self).__init__(args)

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        if self.args.enc_in==7:#default setting
            self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().view(-1))
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().view(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return
