#!/usr/bin/env python
# coding=utf-8
"""Fine-tune and evaluate vinai/phobert-base on UIT Vietnamese Students' Feedback."""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from datasets import ClassLabel, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="vinai/phobert-base",
        metadata={"help": "Model name or local path. Defaults to vinai/phobert-base."},
    )
    dataset_name: str = field(
        default="uitnlp/vietnamese_students_feedback",
        metadata={"help": "Hugging Face dataset name."},
    )
    text_column_name: str = field(default="sentence", metadata={"help": "Input text column name."})
    label_column_name: str = field(
        default="sentiment",
        metadata={"help": "Label column name. Use 'sentiment' or 'topic' for this dataset."},
    )
    max_seq_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length after PhoBERT subword tokenization."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the number of training examples. Useful for quick runs."},
    )
    pad_to_max_length: bool = field(default=True, metadata={"help": "Pad all samples to max_seq_length."})
    mode: str = field(
        default="me",
        metadata={"help": "Adapter mode. Use 'base' for LoRA or 'me' for MELoRA, matching glue_finetune.sh."},
    )
    rank: int = field(default=8, metadata={"help": "LoRA rank, or mini-LoRA rank when using MELoRA."})
    l_num: int = field(default=2, metadata={"help": "Number of mini-LoRAs for MELoRA."})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_bias: str = field(default="none", metadata={"help": "Bias type for LoRA/MELoRA."})
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["query", "value"],
        metadata={"help": "Target module names for LoRA/MELoRA."},
    )
    wandb_project: str = field(default="", metadata={"help": "Weights & Biases project name."})
    wandb_watch: str = field(default="", metadata={"help": "Weights & Biases watch setting."})
    wandb_log_model: str = field(default="", metadata={"help": "Weights & Biases model logging setting."})
    prediction_details_file: str = field(
        default="test_predictions_full.jsonl",
        metadata={"help": "JSONL file name for per-sample predictions with ground truth and original fields."},
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for models/datasets."})


def get_label_names(raw_datasets: DatasetDict, label_column_name: str) -> List[str]:
    label_feature = raw_datasets["train"].features[label_column_name]
    if isinstance(label_feature, ClassLabel):
        return list(label_feature.names)

    labels = raw_datasets["train"].unique(label_column_name)
    labels.sort()
    return [str(label) for label in labels]


def compute_macro_f1(predictions: np.ndarray, labels: np.ndarray, num_labels: int) -> float:
    f1_scores = []
    for label_id in range(num_labels):
        true_positive = np.sum((predictions == label_id) & (labels == label_id))
        false_positive = np.sum((predictions == label_id) & (labels != label_id))
        false_negative = np.sum((predictions != label_id) & (labels == label_id))

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)

    return float(np.mean(f1_scores))


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    if model_args.wandb_project:
        os.environ["WANDB_PROJECT"] = model_args.wandb_project
    if model_args.wandb_watch:
        os.environ["WANDB_WATCH"] = model_args.wandb_watch
    if model_args.wandb_log_model:
        os.environ["WANDB_LOG_MODEL"] = model_args.wandb_log_model

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )

    set_seed(training_args.seed)

    raw_datasets = load_dataset(model_args.dataset_name, cache_dir=model_args.cache_dir)
    for split, dataset in raw_datasets.items():
        missing_columns = [
            column
            for column in (model_args.text_column_name, model_args.label_column_name)
            if column not in dataset.column_names
        ]
        if missing_columns:
            raise ValueError(f"Split '{split}' is missing columns: {missing_columns}")

    if model_args.max_train_samples is not None:
        max_train_samples = min(len(raw_datasets["train"]), model_args.max_train_samples)
        raw_datasets["train"] = raw_datasets["train"].select(range(max_train_samples))

    label_names = get_label_names(raw_datasets, model_args.label_column_name)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(label_names),
        id2label=id_to_label,
        label2id=label_to_id,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if "me" in model_args.mode:
        try:
            from peft import MELoraConfig
        except ImportError as exc:
            raise ImportError(
                "MELoraConfig was not found in the active peft package. "
                "Install this repo's PEFT fork first: cd peft-0.5.0 && pip install -e ."
            ) from exc

        logger.info("*** MELora !!! ***")
        peft_config = MELoraConfig(
            r=[model_args.rank] * model_args.l_num,
            lora_alpha=[model_args.lora_alpha] * model_args.l_num,
            target_modules=model_args.target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            mode=model_args.mode,
            task_type="SEQ_CLS",
            modules_to_save=["classifier"],
        )
    elif "base" in model_args.mode:
        logger.info("*** Just Lora !!! ***")
        peft_config = LoraConfig(
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="SEQ_CLS",
            modules_to_save=["classifier"],
        )
    else:
        raise ValueError(f"Unknown mode {model_args.mode}")

    model = get_peft_model(model, peft_config)

    padding = "max_length" if model_args.pad_to_max_length else False

    def preprocess_function(examples: Dict[str, List[str]]):
        result = tokenizer(
            examples[model_args.text_column_name],
            padding=padding,
            max_length=model_args.max_seq_length,
            truncation=True,
        )
        result["labels"] = examples[model_args.label_column_name]
        return result

    remove_columns = raw_datasets["train"].column_names
    with training_args.main_process_first(desc="dataset tokenization"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing feedback sentences",
        )

    def compute_metrics(predictions_and_labels):
        predictions, labels = predictions_and_labels
        predictions = np.argmax(predictions, axis=1)
        accuracy = float(np.mean(predictions == labels))
        macro_f1 = compute_macro_f1(predictions, labels, len(label_names))
        return {"accuracy": accuracy, "f1": macro_f1}

    data_collator = default_data_collator if model_args.pad_to_max_length else DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predictions, labels, metrics = trainer.predict(tokenized_datasets["test"], metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_details_file = os.path.join(training_args.output_dir, model_args.prediction_details_file)
        if trainer.is_world_process_zero():
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(output_details_file, "w", encoding="utf-8") as writer:
                for index, (sample, prediction, label) in enumerate(zip(raw_datasets["test"], predictions, labels)):
                    record = dict(sample)
                    record.update(
                        {
                            "sample_index": index,
                            "ground_truth": id_to_label[int(label)],
                            "prediction": id_to_label[int(prediction)],
                            "correct": bool(prediction == label),
                        }
                    )
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
