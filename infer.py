"""
Inference script for mental health classification models.

Usage:
    # Single checkpoint folder
    python infer.py --ckpt checkpoints/classification_dep_cls_wobg_clip_... --input /path/to/Video1_1.npy

    # Multiple checkpoints (multi-task)
    python infer.py --ckpt ckpt1 ckpt2 ckpt3 --input /path/to/npy_folder

    # Input can be a single .npy file or a folder of .npy files
"""

import os
import re
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

# ── model registry (mirrors exp_basic.py) ──────────────────────────────────
from models import (
    Autoformer, Transformer, TimesNet, DLinear, FEDformer,
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST,
    MICN, Crossformer, FiLM, iTransformer, Koopa, TiDE, FreTS,
    TimeMixer, TSMixer, SegRNN, TemporalFusionTransformer, SCINet,
    PAttn, TimeXer, WPMixer, MultiPatchFormer, KANAD, Performer, Linearformer,
    Nonstationary_Transformer, MambaSimple,
)

MODEL_DICT = {
    'TimesNet': TimesNet, 'Autoformer': Autoformer, 'Transformer': Transformer,
    'Nonstationary_Transformer': Nonstationary_Transformer, 'DLinear': DLinear,
    'FEDformer': FEDformer, 'Informer': Informer, 'LightTS': LightTS,
    'Reformer': Reformer, 'ETSformer': ETSformer, 'PatchTST': PatchTST,
    'Pyraformer': Pyraformer, 'MICN': MICN, 'Crossformer': Crossformer,
    'FiLM': FiLM, 'iTransformer': iTransformer, 'Koopa': Koopa, 'TiDE': TiDE,
    'FreTS': FreTS, 'MambaSimple': MambaSimple, 'TimeMixer': TimeMixer,
    'TSMixer': TSMixer, 'SegRNN': SegRNN,
    'TemporalFusionTransformer': TemporalFusionTransformer,
    'SCINet': SCINet, 'PAttn': PAttn, 'TimeXer': TimeXer, 'WPMixer': WPMixer,
    'MultiPatchFormer': MultiPatchFormer, 'KANAD': KANAD,
    'Performer': Performer, 'Linearformer': Linearformer,
}

CLASS_NAMES = ['No', 'Yes']

# ── setting name parser ─────────────────────────────────────────────────────
ENC_IN_RE = re.compile(r'enc(\d+)')


def parse_setting(folder_name):
    """Parse checkpoint folder name into args namespace using key-value extraction."""
    s = folder_name

    def extract(pattern, cast=str, default=None):
        m = re.search(pattern, s)
        return cast(m.group(1)) if m else default

    # Identify model by trying each known model name (longest first to avoid prefix clash)
    model = None
    for name in sorted(MODEL_DICT.keys(), key=len, reverse=True):
        if f'_{name}_' in s:
            model = name
            break
    if model is None:
        raise ValueError(f"Cannot identify model in: {folder_name}")

    enc_in = extract(r'enc(\d+)', int, 64)

    args = SimpleNamespace(
        task_name='classification',
        model=model,
        features=extract(r'_ft([^_]+)_sl'),
        seq_len=extract(r'_sl(\d+)_', int, 256),
        label_len=extract(r'_ll(\d+)_', int, 48),
        pred_len=extract(r'_pl(\d+)_', int, 0),
        d_model=extract(r'_dm(\d+)_', int, 128),
        n_heads=extract(r'_nh(\d+)_', int, 8),
        e_layers=extract(r'_el(\d+)_', int, 3),
        d_layers=extract(r'_dl(\d+)_', int, 1),
        d_ff=extract(r'_df(\d+)_', int, 256),
        expand=extract(r'_expand(\d+)_', int, 2),
        d_conv=extract(r'_dc(\d+)_', int, 4),
        factor=extract(r'_fc(\d+)_', int, 1),
        embed=extract(r'_eb([^_]+)_dt', str, 'timeF'),
        distil=extract(r'_dt([^_]+)_', str, 'True').lower() == 'true',
        enc_in=enc_in,
        dec_in=7,
        c_out=7,
        num_class=2,
        dropout=0.1,
        activation='gelu',
        output_attention=False,
        moving_avg=25,
        channel_independence=1,
        decomp_method='moving_avg',
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method='avg',
        seg_len=48,
        top_k=3,
        num_kernels=6,
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        use_gpu=True,
        gpu=0,
        gpu_type='cuda',
        use_multi_gpu=False,
        devices='0,1',
        freq='h',
        mask_rate=0.25,
        loss='MSE',
        patch_len=16,
        stride=8,
    )
    return args


# ── feature loading ─────────────────────────────────────────────────────────

def load_npy_input(path):
    """Load a single npy file or all npy files in a folder.
    Returns dict {name: np.ndarray}."""
    if os.path.isfile(path):
        return {os.path.basename(path): np.load(path)}
    elif os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith('.npy'))
        return {f: np.load(os.path.join(path, f)) for f in files}
    else:
        raise FileNotFoundError(f"Input not found: {path}")


def resample(arr, target_len):
    """Resample (T, F) array to (target_len, F)."""
    t = torch.from_numpy(arr.astype(np.float32))  # (T, F)
    if t.shape[0] == target_len:
        return t
    r = F.interpolate(t.T.unsqueeze(0), size=target_len, mode='linear', align_corners=False)
    return r.squeeze(0).T  # (target_len, F)


def prepare_tensor(arr, seq_len, enc_in, device):
    """arr: (T, F) -> batch tensor (1, seq_len, enc_in) + padding_mask."""
    t = resample(arr, seq_len)           # (seq_len, F)
    # feature subsampling: pick enc_in dims with fixed seed
    if t.shape[1] > enc_in:
        np.random.seed(42)
        idx = np.random.choice(t.shape[1], size=enc_in, replace=False)
        t = t[:, idx]
    elif t.shape[1] < enc_in:
        pad = torch.zeros(seq_len, enc_in - t.shape[1])
        t = torch.cat([t, pad], dim=1)
    batch = t.unsqueeze(0).float().to(device)        # (1, seq_len, enc_in)
    mask = torch.ones(1, seq_len).float().to(device)  # no padding
    return batch, mask


# ── model loader ─────────────────────────────────────────────────────────────

def load_model(ckpt_dir, device):
    folder = os.path.basename(ckpt_dir.rstrip('/'))
    args = parse_setting(folder)
    model = MODEL_DICT[args.model].Model(args).float().to(device)
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pth')
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, args


# ── inference ────────────────────────────────────────────────────────────────

def infer_one(model, args, arr, device):
    """Run inference on one (T, F) array. Returns (pred_class, probs)."""
    x, mask = prepare_tensor(arr, args.seq_len, args.enc_in, device)
    with torch.no_grad():
        logits = model(x, mask, None, None)   # (1, num_class)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs


def run_inference(ckpt_dirs, input_path, device):
    arrays = load_npy_input(input_path)
    results = {}  # name -> list of per-ckpt dicts

    for ckpt_dir in ckpt_dirs:
        folder = os.path.basename(ckpt_dir.rstrip('/'))
        print(f"\n[{folder}]")
        try:
            model, args = load_model(ckpt_dir, device)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        for name, arr in arrays.items():
            pred, probs = infer_one(model, args, arr, device)
            entry = {
                'ckpt': folder,
                'pred': pred,
                'pred_label': CLASS_NAMES[pred],
                'prob_No': float(probs[0]),
                'prob_Yes': float(probs[1]),
            }
            results.setdefault(name, []).append(entry)
            print(f"  {name}: {CLASS_NAMES[pred]}  (No={probs[0]:.3f}, Yes={probs[1]:.3f})")

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', nargs='+', required=True,
                        help='Checkpoint folder(s)')
    parser.add_argument('--input', required=True,
                        help='Single .npy file or folder of .npy files')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu') if args.cpu or not torch.cuda.is_available() \
        else torch.device(f'cuda:{args.gpu}')
    print(f"Device: {device}")

    run_inference(args.ckpt, args.input, device)


# ── demo / test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if '--ckpt' in sys.argv:
        main()
    else:
        # Demo: use best model per task, test on a random sample from preprocess
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}\n")

        PREPROCESS = '/data2/lx/mental/preprocess/output/wobg_clip'
        sample_files = sorted(f for f in os.listdir(PREPROCESS) if f.endswith('.npy'))
        test_file = os.path.join(PREPROCESS, sample_files[0])
        print(f"Test input: {test_file}")
        arr = np.load(test_file)
        print(f"Shape: {arr.shape}\n")

        # Best checkpoint per task (from coo.py results)
        BEST_CKPTS = [
            'checkpoints/classification_dep_cls_wobg_clip_seq256_enc64_lr0.0002_Informer_Mental_ftwobg_clip_sl256_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_seq256_wobg_clip_lr0.0002_0',
            'checkpoints/classification_anx_cls_wobg_clip_seq256_enc64_lr0.0002_Reformer_Mental_ftwobg_clip_sl256_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_seq256_wobg_clip_lr0.0002_0',
            'checkpoints/classification_sui_cls_wobg_clip_seq256_enc64_lr0.0002_FiLM_Mental_ftwobg_clip_sl256_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_seq256_wobg_clip_lr0.0002_0',
            'checkpoints/classification_ovr_cls_wobg_clip_seq256_enc64_lr0.0002_DLinear_Mental_ftwobg_clip_sl256_ll48_pl0_dm128_nh8_el3_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_Exp_seq256_wobg_clip_lr0.0002_0',
        ]

        TASK_LABELS = ['depression', 'anxiety', 'suiside', 'overall']

        print("=" * 60)
        print(f"{'Task':<12} {'Pred':<6} {'P(No)':<8} {'P(Yes)':<8} Model")
        print("=" * 60)
        for ckpt_dir, task in zip(BEST_CKPTS, TASK_LABELS):
            if not os.path.exists(os.path.join(ckpt_dir, 'checkpoint.pth')):
                print(f"{task:<12} MISSING: {ckpt_dir}")
                continue
            try:
                model, margs = load_model(ckpt_dir, device)
                pred, probs = infer_one(model, margs, arr, device)
                print(f"{task:<12} {CLASS_NAMES[pred]:<6} {probs[0]:<8.3f} {probs[1]:<8.3f} {margs.model}")
            except Exception as e:
                print(f"{task:<12} ERROR: {e}")
        print("=" * 60)
