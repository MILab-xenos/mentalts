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
    
warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

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

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='TEST')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, label, padding_mask) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             padding_mask = padding_mask.float().to(self.device)
    #             label = label.to(self.device)

    #             outputs = self.model(batch_x, padding_mask, None, None)

    #             preds.append(outputs.detach())
    #             trues.append(label)

    #     preds = torch.cat(preds, 0)
    #     trues = torch.cat(trues, 0)
    #     print('test shape:', preds.shape, trues.shape)

    #     probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
    #     predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    #     trues = trues.flatten().cpu().numpy()
    #     accuracy = cal_accuracy(predictions, trues)

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     print('accuracy:{}'.format(accuracy))
    #     file_name='result_classification.txt'
    #     f = open(os.path.join(folder_path,file_name), 'a')
    #     f.write(setting + "  \n")
    #     f.write('accuracy:{}'.format(accuracy))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()
    #     return
    def test(self, setting, test=0):
        from sklearn.metrics import (
            precision_recall_fscore_support, roc_auc_score, average_precision_score,
            confusion_matrix, classification_report
        )
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import os

        def plot_confusion_matrix(cm, class_names, save_path):
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
            plt.close()

        def plot_roc_pr_curves(y_true, y_score, n_classes, class_names, save_path):
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc, precision_recall_curve

            # 强制转 numpy
            y_true = np.array(y_true)
            y_score = np.array(y_score)

            # === 二分类 ===
            if n_classes == 2:
                if y_score.shape[1] == 1:  # sigmoid 输出情况
                    y_score = np.hstack([1 - y_score, y_score])
                y_true_bin = label_binarize(y_true, classes=[0, 1])

                fpr, tpr, _ = roc_curve(y_true_bin[:, 0], y_score[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.figure(figsize=(7, 6))
                plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC={roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', lw=1)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'roc_curve.png'))
                plt.close()

                precision, recall, _ = precision_recall_curve(y_true_bin[:, 0], y_score[:, 1])
                pr_auc = auc(recall, precision)
                plt.figure(figsize=(7, 6))
                plt.plot(recall, precision, lw=2, label=f'PR (AUC={pr_auc:.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'pr_curve.png'))
                plt.close()

            # === 多分类 ===
            else:
                y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
                plt.figure(figsize=(7, 6))
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC={roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', lw=1)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curves')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'roc_curve.png'))
                plt.close()

                plt.figure(figsize=(7, 6))
                for i in range(n_classes):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
                    pr_auc = auc(recall, precision)
                    plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AUC={pr_auc:.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curves')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, 'pr_curve.png'))
                plt.close()

        # ==== 数据准备 ====
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, padding_mask, None, None)
                preds.append(outputs.detach().cpu())
                trues.append(label.detach().cpu())

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).numpy()
        trues = trues.flatten().numpy()
        probs_np = probs.numpy()  # ✅ 全部转 numpy

        # ==== 计算指标 ====
        from utils.tools import cal_accuracy
        accuracy = cal_accuracy(predictions, trues)
        print(f'accuracy: {accuracy:.4f}')

        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(trues, predictions, average='macro')
        weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(trues, predictions, average='weighted')

        try:
            macro_rocauc = roc_auc_score(trues, probs_np[:,1], multi_class='ovr', average='macro')
            weighted_rocauc = roc_auc_score(trues, probs_np[:,1], multi_class='ovr', average='weighted')
            macro_prauc = average_precision_score(trues, probs_np[:,1], average='macro')
            weighted_prauc = average_precision_score(trues, probs_np[:,1], average='weighted')
        except Exception as e:
            print(f"ROC/PR AUC calculation failed: {e}")
            macro_rocauc = weighted_rocauc = macro_prauc = weighted_prauc = np.nan

        cm = confusion_matrix(trues, predictions)
        class_names = getattr(test_data, 'class_names', [f'class_{i}' for i in np.unique(trues)])
        plot_confusion_matrix(cm, class_names, folder_path)
        plot_roc_pr_curves(trues, probs_np, len(class_names), class_names, folder_path)

        # ==== 保存结果 ====
        with open(os.path.join(folder_path, 'result_classification.txt'), 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro Precision: {macro_p:.4f}, Recall: {macro_r:.4f}, F1: {macro_f1:.4f}\n")
            f.write(f"Weighted Precision: {weighted_p:.4f}, Recall: {weighted_r:.4f}, F1: {weighted_f1:.4f}\n")
            f.write(f"Macro ROC-AUC: {macro_rocauc:.4f}, PR-AUC: {macro_prauc:.4f}\n")
            f.write(f"Weighted ROC-AUC: {weighted_rocauc:.4f}, PR-AUC: {weighted_prauc:.4f}\n\n")
            f.write(classification_report(trues, predictions, target_names=class_names,digits=6))
            f.write("\n\n")

        print('Detailed metrics and plots saved to:', folder_path)