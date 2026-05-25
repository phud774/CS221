#!/bin/bash

export WANDB_MODE=offline

run(){
  mode=$1
  rank=$2
  l_num=$3
  seed=42
  learning_rate=4e-4
  num_train_epochs=40
  max_train_samples=5000
  batch_size=64
  lora_alpha=16
  lora_dropout=0.05
  target_modules="query value"
  label_column_name=upos
  validation_ratio=0.1
  test_ratio=0.1
  prediction_details_file="test_predictions_full.jsonl"
  wandb_project=project_name
  model_name_or_path=vinai/phobert-base
  model_short_name=phobert-base
  wandb_run_name=${model_short_name}-udd-${label_column_name}-${mode}-r-${rank}-n-${l_num}-alpha-${lora_alpha}-seed-${seed}-bs-${batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}-train-${max_train_samples}
  exp_dir=./${model_short_name}-udd-${label_column_name}/${wandb_run_name}

  python ./run_udd_phobert.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset_name undertheseanlp/UDD-v0.1 \
  --tokens_column_name tokens \
  --label_column_name ${label_column_name} \
  --validation_ratio ${validation_ratio} \
  --test_ratio ${test_ratio} \
  --output_dir ${exp_dir}/model \
  --do_train \
  --do_eval \
  --do_predict \
  --mode ${mode} \
  --rank ${rank} \
  --l_num ${l_num} \
  --lora_alpha ${lora_alpha} \
  --lora_dropout ${lora_dropout} \
  --lora_bias none \
  --target_modules ${target_modules} \
  --prediction_details_file ${prediction_details_file} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end true \
  --metric_for_best_model f1 \
  --greater_is_better true \
  --max_seq_length 256 \
  --max_train_samples ${max_train_samples} \
  --per_device_train_batch_size ${batch_size} \
  --per_device_eval_batch_size ${batch_size} \
  --learning_rate ${learning_rate} \
  --num_train_epochs ${num_train_epochs} \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --logging_steps 20 \
  --seed ${seed} \
  --wandb_project ${wandb_project} \
  --report_to wandb \
  --run_name ${wandb_run_name} \
  --overwrite_output_dir
}

# LoRA baseline, same config style as students_feedback_phobert.sh.
run "base" "8" "1"

# MELoRA run, same config style as students_feedback_phobert.sh.
run "me" "8" "2"
