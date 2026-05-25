#!/usr/bin/env python
# coding=utf-8
"""Fine-tune and evaluate RoBERTa LoRA/MELoRA on tner/bc5cdr."""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from datasets import ClassLabel, DatasetDict, Sequence, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


logger = logging.getLogger(__name__)

BC5CDR_LABELS = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]
BC5CDR_DATA_URL = "https://huggingface.co/datasets/tner/bc5cdr/raw/main/dataset"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="FacebookAI/roberta-base", metadata={"help": "Model name or path."})
    dataset_name: str = field(default="tner/bc5cdr", metadata={"help": "Hugging Face dataset name."})
    tokens_column_name: str = field(default="tokens", metadata={"help": "Token column name."})
    tags_column_name: str = field(default="tags", metadata={"help": "NER tag column name."})
    max_seq_length: int = field(default=256, metadata={"help": "Maximum tokenized sequence length."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Limit training examples."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Limit validation examples."})
    max_predict_samples: Optional[int] = field(default=None, metadata={"help": "Limit test examples."})
    label_all_tokens: bool = field(
        default=False,
        metadata={"help": "If true, label every subword. If false, only label the first subword of each word."},
    )
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


def get_label_list(raw_datasets: DatasetDict, tags_column_name: str) -> List[str]:
    feature = raw_datasets["train"].features[tags_column_name]
    if isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel):
        return list(feature.feature.names)

    unique_ids = set()
    for split in raw_datasets:
        for tags in raw_datasets[split][tags_column_name]:
            unique_ids.update(int(tag) for tag in tags)

    if unique_ids == set(range(len(BC5CDR_LABELS))):
        return BC5CDR_LABELS
    return [str(index) for index in sorted(unique_ids)]


def load_bc5cdr_dataset(dataset_name: str, cache_dir: Optional[str]) -> DatasetDict:
    try:
        return load_dataset(dataset_name, cache_dir=cache_dir)
    except RuntimeError as exc:
        message = str(exc)
        if dataset_name != "tner/bc5cdr" or "Dataset scripts are no longer supported" not in message:
            raise

        logger.warning(
            "Falling back to direct JSONL files for %s because its dataset script is no longer supported.",
            dataset_name,
        )
        data_files = {
            "train": f"{BC5CDR_DATA_URL}/train.json",
            "validation": f"{BC5CDR_DATA_URL}/valid.json",
            "test": f"{BC5CDR_DATA_URL}/test.json",
        }
        return load_dataset("json", data_files=data_files, cache_dir=cache_dir)


def b_to_i(label_id: int, label_list: List[str]) -> int:
    label = label_list[label_id]
    if not label.startswith("B-"):
        return label_id
    i_label = f"I-{label[2:]}"
    return label_list.index(i_label) if i_label in label_list else label_id


def tokenize_and_align_labels(examples: Dict[str, List[List[str]]], tokenizer, model_args: ModelArguments):
    tokenized_inputs = tokenizer(
        examples[model_args.tokens_column_name],
        is_split_into_words=True,
        padding=False,
        truncation=True,
        max_length=model_args.max_seq_length,
    )

    labels = []
    for batch_index, word_tags in enumerate(examples[model_args.tags_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(int(word_tags[word_id]))
            else:
                label_ids.append(b_to_i(int(word_tags[word_id]), model_args.label_list) if model_args.label_all_tokens else -100)
            previous_word_id = word_id
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def align_predictions_to_words(sample, prediction, aligned_label_ids, label_list, tokens_column_name, tags_column_name):
    tokens = []
    pred_index = 0
    correct_count = 0

    for word, gold_id in zip(sample[tokens_column_name], sample[tags_column_name]):
        while pred_index < len(aligned_label_ids) and aligned_label_ids[pred_index] == -100:
            pred_index += 1
        if pred_index >= len(aligned_label_ids):
            tokens.append(
                {
                    "word": word,
                    "ground_truth": label_list[int(gold_id)],
                    "prediction": None,
                    "correct": False,
                    "truncated": True,
                }
            )
            continue

        predicted_label = label_list[int(prediction[pred_index])]
        gold_label = label_list[int(gold_id)]
        is_correct = predicted_label == gold_label
        correct_count += int(is_correct)
        tokens.append(
            {
                "word": word,
                "ground_truth": gold_label,
                "prediction": predicted_label,
                "correct": is_correct,
                "truncated": False,
            }
        )
        pred_index += 1

    return tokens, correct_count


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

    raw_datasets = load_bc5cdr_dataset(model_args.dataset_name, model_args.cache_dir)
    for split, dataset in raw_datasets.items():
        missing_columns = [
            column
            for column in (model_args.tokens_column_name, model_args.tags_column_name)
            if column not in dataset.column_names
        ]
        if missing_columns:
            raise ValueError(f"Split '{split}' is missing columns: {missing_columns}")

    label_list = get_label_list(raw_datasets, model_args.tags_column_name)
    model_args.label_list = label_list
    id_to_label = {index: label for index, label in enumerate(label_list)}
    label_to_id = {label: index for index, label in id_to_label.items()}

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id_to_label,
        label2id=label_to_id,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
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
            task_type="TOKEN_CLS",
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
            task_type="TOKEN_CLS",
            modules_to_save=["classifier"],
        )
    else:
        raise ValueError(f"Unknown mode {model_args.mode}")

    model = get_peft_model(model, peft_config)

    with training_args.main_process_first(desc="dataset tokenization"):
        tokenized_datasets = raw_datasets.map(
            lambda examples: tokenize_and_align_labels(examples, tokenizer, model_args),
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing and aligning BC5CDR labels",
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
        from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

        predictions, labels = predictions_and_labels
        predictions = np.argmax(predictions, axis=2)
        true_predictions = []
        true_labels = []
        for prediction, label in zip(predictions, labels):
            filtered_predictions = []
            filtered_labels = []
            for predicted_id, label_id in zip(prediction, label):
                if label_id == -100:
                    continue
                filtered_predictions.append(label_list[int(predicted_id)])
                filtered_labels.append(label_list[int(label_id)])
            true_predictions.append(filtered_predictions)
            true_labels.append(filtered_labels)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
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

        predictions = np.argmax(predictions, axis=2)
        raw_test_dataset = raw_datasets["test"]
        if model_args.max_predict_samples is not None:
            raw_test_dataset = raw_test_dataset.select(range(min(len(raw_test_dataset), model_args.max_predict_samples)))

        output_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        output_details_file = os.path.join(training_args.output_dir, model_args.prediction_details_file)
        if trainer.is_world_process_zero():
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(output_predictions_file, "w", encoding="utf-8") as writer:
                for sample, prediction, label in zip(raw_test_dataset, predictions, labels):
                    tokens, _ = align_predictions_to_words(
                        sample,
                        prediction,
                        label,
                        label_list,
                        model_args.tokens_column_name,
                        model_args.tags_column_name,
                    )
                    for token in tokens:
                        writer.write(
                            f"{token['word']}\t{token['ground_truth']}\t{token['prediction']}\t{token['correct']}\n"
                        )
                    writer.write("\n")

            with open(output_details_file, "w", encoding="utf-8") as writer:
                for index, (sample, prediction, label) in enumerate(zip(raw_test_dataset, predictions, labels)):
                    tokens, correct_count = align_predictions_to_words(
                        sample,
                        prediction,
                        label,
                        label_list,
                        model_args.tokens_column_name,
                        model_args.tags_column_name,
                    )
                    predicted_count = sum(not token["truncated"] for token in tokens)
                    record = dict(sample)
                    record.update(
                        {
                            "sample_index": index,
                            "tokens": tokens,
                            "num_tokens": len(tokens),
                            "num_predicted_tokens": predicted_count,
                            "num_correct_tokens": correct_count,
                            "token_accuracy": correct_count / predicted_count if predicted_count else 0.0,
                        }
                    )
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
