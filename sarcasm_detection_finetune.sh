#!/bin/bash

export WANDB_MODE=offline

run(){
  learning_rate=4e-4
  num_train_epochs=40
  per_device_train_batch_size=64
  rank=$1
  l_num=$2
  seed=42
  lora_alpha=16
  target_modules="query value"
  mode=$3
  lora_dropout=0.05
  lora_bias=none
  max_seq_length=256
  validation_ratio=0.1
  test_ratio=0.1
  wandb_project=project_name
  wandb_run_name=roberta-lora-${mode}-sarcasm-headlines-r-${rank}-n-${l_num}-alpha-${lora_alpha}-seed-${seed}-bs-${per_device_train_batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}

  exp_dir=../roberta_sarcasm_reproduce/${wandb_run_name}

  CUDA_VISIBLE_DEVICES=0 python ./run_sarcasm_detection_lora.py \
  --model_name_or_path FacebookAI/roberta-base \
  --dataset_name Heschmat/news-headlines-dataset-sarcasm-detection \
  --text_column_name headline \
  --label_column_name is_sarcastic \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length ${max_seq_length} \
  --validation_ratio ${validation_ratio} \
  --test_ratio ${test_ratio} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_train_batch_size} \
  --load_best_model_at_end true \
  --metric_for_best_model f1 \
  --greater_is_better true \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --weight_decay 0.1 \
  --warmup_ratio 0.06 \
  --logging_steps 10 \
  --seed ${seed} \
  --wandb_project ${wandb_project} \
  --lora_alpha ${lora_alpha} \
  --lora_dropout ${lora_dropout} \
  --lora_bias ${lora_bias} \
  --target_modules ${target_modules} \
  --rank ${rank} \
  --l_num ${l_num} \
  --mode ${mode} \
  --output_dir ${exp_dir}/model \
  --logging_dir ${exp_dir}/log \
  --run_name ${wandb_run_name} \
  --report_to wandb \
  --overwrite_output_dir
}

# LoRA baseline, same style as glue_finetune.sh.
run "8" "1" "base"

# MELoRA run, same style as glue_finetune.sh.
run "8" "2" "me"
