#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name=iTransformer

seq_lens=(256)
enc_ins=(64)
featuress=(wobg_clip)
lrs=(0.0002)

# 遍历所有组合
for seq_len in "${seq_lens[@]}"
do
  for enc_in in "${enc_ins[@]}"
  do
    for features in "${featuress[@]}"
    do
      for lr in "${lrs[@]}"
      do
        echo "===== Running seq_len=$seq_len, enc_in=$enc_in, features=$features, lr=$lr ====="
        python -u run.py \
          --task_name classification \
          --is_training 0 \
          --root_path '/data2/lx/mental/preprocess/output' \
          --model_id "dep_cls_${features}_seq${seq_len}_enc${enc_in}_lr${lr}" \
          --model $model_name \
          --data Mental \
          --features $features \
          --target depression \
          --seq_len $seq_len \
          --enc_in $enc_in \
          --e_layers 3 \
          --batch_size 16 \
          --d_model 2048 \
          --d_ff 256 \
          --top_k 3 \
          --des "Exp_seq${seq_len}_${features}_lr${lr}" \
          --itr 1 \
          --learning_rate $lr \
          --train_epochs 100 \
          --patience 10
      done
    done
  done
done