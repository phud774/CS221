#!/usr/bin/env python
# coding=utf-8
"""Fine-tune and evaluate RoBERTa LoRA/MELoRA on headline sarcasm detection."""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from datasets import DatasetDict, load_dataset
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
        default="FacebookAI/roberta-base",
        metadata={"help": "Model name or local path."},
    )
    dataset_name: str = field(
        default="Heschmat/news-headlines-dataset-sarcasm-detection",
        metadata={"help": "Hugging Face dataset name."},
    )
    text_column_name: str = field(default="headline", metadata={"help": "Input text column name."})
    label_column_name: str = field(default="is_sarcastic", metadata={"help": "Label column name."})
    max_seq_length: int = field(default=256, metadata={"help": "Maximum tokenized sequence length."})
    pad_to_max_length: bool = field(default=True, metadata={"help": "Pad all samples to max_seq_length."})
    validation_ratio: float = field(
        default=0.1,
        metadata={"help": "Validation ratio used when the dataset has no validation split."},
    )
    test_ratio: float = field(default=0.1, metadata={"help": "Test ratio used when the dataset has no test split."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the number of training examples. Useful for quick runs."},
    )
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit validation examples."})
    max_predict_samples: Optional[int] = field(default=None, metadata={"help": "Limit test examples."})
    mode: str = field(default="me", metadata={"help": "Use 'base' for LoRA or 'me' for MELoRA."})
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
    prediction_details_file: str = field(
        default="test_predictions_full.jsonl",
        metadata={"help": "JSONL file name for per-sample predictions."},
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for models/datasets."})


def prepare_splits(raw_datasets: DatasetDict, seed: int, validation_ratio: float, test_ratio: float) -> DatasetDict:
    if "validation" in raw_datasets and "test" in raw_datasets:
        return raw_datasets

    if "train" not in raw_datasets:
        raise ValueError("Dataset must contain a train split when validation/test splits are absent.")
    if validation_ratio <= 0 or test_ratio <= 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be positive and sum to less than 1.")

    heldout_ratio = validation_ratio + test_ratio
    first_split = raw_datasets["train"].train_test_split(test_size=heldout_ratio, seed=seed, stratify_by_column=None)
    test_fraction_of_heldout = test_ratio / heldout_ratio
    second_split = first_split["test"].train_test_split(
        test_size=test_fraction_of_heldout,
        seed=seed,
        stratify_by_column=None,
    )
    return DatasetDict(
        {
            "train": first_split["train"],
            "validation": second_split["train"],
            "test": second_split["test"],
        }
    )


def compute_binary_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    true_positive = np.sum((predictions == 1) & (labels == 1))
    false_positive = np.sum((predictions == 1) & (labels == 0))
    false_negative = np.sum((predictions == 0) & (labels == 1))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0


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
    logger.setLevel(training_args.get_process_log_level())

    if model_args.wandb_project:
        os.environ["WANDB_PROJECT"] = model_args.wandb_project

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
    raw_datasets = prepare_splits(raw_datasets, training_args.seed, model_args.validation_ratio, model_args.test_ratio)

    for split, dataset in raw_datasets.items():
        missing_columns = [
            column
            for column in (model_args.text_column_name, model_args.label_column_name)
            if column not in dataset.column_names
        ]
        if missing_columns:
            raise ValueError(f"Split '{split}' is missing columns: {missing_columns}")

    label_names = ["not_sarcastic", "sarcastic"]
    id_to_label = {index: label for index, label in enumerate(label_names)}
    label_to_id = {label: index for index, label in id_to_label.items()}

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
        use_fast=True,
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
        result["labels"] = [int(label) for label in examples[model_args.label_column_name]]
        return result

    with training_args.main_process_first(desc="dataset tokenization"):
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing headlines",
        )

    train_dataset = tokenized_datasets["train"] if training_args.do_train else None
    eval_dataset = tokenized_datasets["validation"] if training_args.do_eval else None
    predict_dataset = tokenized_datasets["test"] if training_args.do_predict else None

    if train_dataset is not None and model_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), model_args.max_train_samples)))
    if eval_dataset is not None and model_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), model_args.max_eval_samples)))
    if predict_dataset is not None and model_args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(min(len(predict_dataset), model_args.max_predict_samples)))

    def compute_metrics(predictions_and_labels):
        predictions, labels = predictions_and_labels
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": float(np.mean(predictions == labels)),
            "f1": compute_binary_f1(predictions, labels),
        }

    data_collator = default_data_collator if model_args.pad_to_max_length else DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="test")
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        predictions = np.argmax(predictions, axis=1)
        output_details_file = os.path.join(training_args.output_dir, model_args.prediction_details_file)
        raw_test_dataset = raw_datasets["test"]
        if model_args.max_predict_samples is not None:
            raw_test_dataset = raw_test_dataset.select(range(min(len(raw_test_dataset), model_args.max_predict_samples)))
        if trainer.is_world_process_zero():
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(output_details_file, "w", encoding="utf-8") as writer:
                for index, (sample, prediction, label) in enumerate(zip(raw_test_dataset, predictions, labels)):
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
