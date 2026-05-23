#!/bin/bash

export WANDB_MODE=offline

run(){
  mode=$1
  rank=$2
  l_num=$3
  seed=42
  learning_rate=1e-4
  num_train_epochs=40
  batch_size=64
  lora_alpha=16
  lora_dropout=0.05
  target_modules="query value"
  prediction_details_file="test_predictions_full.jsonl"
  wandb_project=project_name
  wandb_run_name=phobert-base-vietmed-ner-${mode}-r-${rank}-n-${l_num}-alpha-${lora_alpha}-seed-${seed}-bs-${batch_size}-lr-${learning_rate}-epochs-${num_train_epochs}
  exp_dir=./phobert-base-vietmed-ner/${wandb_run_name}

  python ./run_vietmed_ner_phobert.py \
  --model_name_or_path vinai/phobert-base \
  --dataset_name leduckhai/VietMed-NER \
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

# LoRA baseline, same meaning as glue_finetune.sh's mode=base.
run "base" "8" "1"

# MELoRA run, same style as glue_finetune.sh's mode=me.
run "me" "8" "2"
