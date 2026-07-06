# Mental Health Time-Series Classification

Binary classification of mental health conditions from video feature embeddings (CLIP/DINO), using transformer-based time-series models.

## Tasks

| Task | Label | Best Model | Accuracy |
|------|-------|-----------|----------|
| depression | result-1 | Informer | 0.841 |
| anxiety | result-5 | Reformer | 0.722 |
| suiside | SDS-1~6 weighted | FiLM | 0.833 |
| overall | any result-1~15 = 1 | DLinear | 0.833 |

## Data

Features in `/data2/lx/mental/preprocess/output/`:

| Folder | Source | Dim |
|--------|--------|-----|
| `wobg_clip` | background-removed CLIP | 512 |
| `src_clip` | original CLIP | 512 |
| `crop_clip` | cropped CLIP | 512 |
| `crop_dino` | cropped DINOv2 | 768 |

Labels from `data.csv` (1027 cases, 70/15/15 train/val/test split, seed=42).

## Training

```bash
# Single model
bash scripts/depclassification/Informer.sh

# All models for a task (parallel)
bash run_dep.sh
bash run.sh        # anxiety
bash run_suiside.sh
bash run_overall.sh
```

Already-completed runs are skipped automatically (checkpoint exists check in `run.py`).

## Inference

```bash
# Demo: run best model per task on one sample
python infer.py

# Single checkpoint + single file
python infer.py --ckpt checkpoints/<setting> --input /path/to/Video1_1.npy

# Multiple tasks + folder of .npy files
python infer.py --ckpt checkpoints/dep_ckpt checkpoints/anx_ckpt --input /path/to/npy_folder/
```

Output: predicted class (`No`/`Yes`) and probabilities for each task.

## Results

```bash
python coo.py          # print all metrics grouped by task, save classification_summary.csv
```

## Project Structure

```
run.py                          # training entry point
infer.py                        # inference entry point
coo.py                          # results aggregation
data_provider/mental.py         # MentalLoader dataset
exp/exp_classification.py       # training / evaluation loop
models/                         # 30+ transformer architectures
scripts/
  depclassification/            # depression training scripts
  anxclassification/            # anxiety
  suisideclassification/        # suicide risk
  overallclassification/        # overall
checkpoints/                    # saved model weights (git-lfs)
test_results/                   # per-run metrics
```
