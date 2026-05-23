#!/usr/bin/env python
# coding=utf-8
"""Fine-tune and evaluate vinai/phobert-base-v2 on leduckhai/VietMed-NER."""

import logging
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from datasets import DatasetDict, load_dataset
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


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="vinai/phobert-base-v2",
        metadata={"help": "Model name or local path. Defaults to vinai/phobert-base-v2."},
    )
    dataset_name: str = field(
        default="leduckhai/VietMed-NER",
        metadata={"help": "Hugging Face dataset name."},
    )
    max_seq_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length after PhoBERT subword tokenization."},
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={"help": "If true, label every subword. If false, only label the first subword of each word."},
    )
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


def normalize_label(label: str) -> str:
    return "O" if label == "0" else label


def b_to_i(label: str) -> str:
    return f"I-{label[2:]}" if label.startswith("B-") else label


def build_label_list(raw_datasets: DatasetDict) -> List[str]:
    labels = set()
    for split in raw_datasets:
        for example_labels in raw_datasets[split]["labels"]:
            labels.update(normalize_label(label) for label in example_labels)

    entity_labels = sorted(label for label in labels if label != "O")
    return ["O"] + entity_labels


def tokenize_and_align_labels(
    examples: Dict[str, List[List[str]]],
    tokenizer,
    label_to_id: Dict[str, int],
    max_seq_length: int,
    label_all_tokens: bool,
):
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    max_tokens_without_specials = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)

    for words, word_labels in zip(examples["words"], examples["labels"]):
        input_ids = []
        aligned_labels = []

        for word, label in zip(words, word_labels):
            label = normalize_label(label)
            pieces = tokenizer.tokenize(str(word))
            if not pieces:
                pieces = [tokenizer.unk_token]

            piece_ids = tokenizer.convert_tokens_to_ids(pieces)
            input_ids.extend(piece_ids)
            aligned_labels.append(label_to_id[label])

            subword_label = label_to_id[b_to_i(label)] if label_all_tokens and b_to_i(label) in label_to_id else -100
            aligned_labels.extend([subword_label] * (len(piece_ids) - 1))

        input_ids = input_ids[:max_tokens_without_specials]
        aligned_labels = aligned_labels[:max_tokens_without_specials]

        special_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=False)
        input_ids_with_specials = tokenizer.build_inputs_with_special_tokens(input_ids)

        labels_with_specials = []
        label_index = 0
        for is_special in special_mask:
            if is_special:
                labels_with_specials.append(-100)
            else:
                labels_with_specials.append(aligned_labels[label_index])
                label_index += 1

        tokenized_inputs["input_ids"].append(input_ids_with_specials)
        tokenized_inputs["attention_mask"].append([1] * len(input_ids_with_specials))
        tokenized_inputs["labels"].append(labels_with_specials)

    return tokenized_inputs


def print_lora_parameters(model):
    trainable_params = 0
    lora_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
            if "lora_" in name:
                lora_params += num_params
            elif "modules_to_save" not in name:
                logger.info("Trainable non-LoRA parameter: %s", name)

    logger.info(
        "lora params: %s || trainable params: %s || all params: %s || trainable%%: %.4f",
        f"{lora_params:,d}",
        f"{trainable_params:,d}",
        f"{all_param:,d}",
        100 * trainable_params / all_param,
    )


def align_predictions_to_words(sample, prediction, aligned_label_ids, label_list):
    tokens = []
    pred_index = 0
    correct_count = 0

    for word, ground_truth in zip(sample["words"], sample["labels"]):
        while pred_index < len(aligned_label_ids) and aligned_label_ids[pred_index] == -100:
            pred_index += 1
        if pred_index >= len(aligned_label_ids):
            tokens.append(
                {
                    "word": word,
                    "ground_truth": normalize_label(ground_truth),
                    "prediction": None,
                    "correct": False,
                    "truncated": True,
                }
            )
            continue

        gold_label = normalize_label(ground_truth)
        predicted_label = label_list[prediction[pred_index]]
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
    label_list = build_label_list(raw_datasets)
    label_to_id = {label: index for index, label in enumerate(label_list)}
    id_to_label = {index: label for label, index in label_to_id.items()}

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
        use_fast=False,
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
    print_lora_parameters(model)

    with training_args.main_process_first(desc="dataset tokenization"):
        tokenized_datasets = raw_datasets.map(
            lambda examples: tokenize_and_align_labels(
                examples,
                tokenizer,
                label_to_id,
                model_args.max_seq_length,
                model_args.label_all_tokens,
            ),
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing and aligning NER labels",
        )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
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
                filtered_predictions.append(label_list[predicted_id])
                filtered_labels.append(label_list[label_id])
            true_predictions.append(filtered_predictions)
            true_labels.append(filtered_labels)

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

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

        predictions = np.argmax(predictions, axis=2)
        output_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        output_details_file = os.path.join(training_args.output_dir, model_args.prediction_details_file)
        if trainer.is_world_process_zero():
            os.makedirs(training_args.output_dir, exist_ok=True)
            with open(output_predictions_file, "w", encoding="utf-8") as writer:
                for sample, prediction, label in zip(raw_datasets["test"], predictions, labels):
                    tokens, _ = align_predictions_to_words(sample, prediction, label, label_list)
                    for token in tokens:
                        writer.write(
                            f"{token['word']}\t{token['ground_truth']}\t{token['prediction']}\t{token['correct']}\n"
                        )
                    writer.write("\n")

            with open(output_details_file, "w", encoding="utf-8") as writer:
                for index, (sample, prediction, label) in enumerate(zip(raw_datasets["test"], predictions, labels)):
                    tokens, correct_count = align_predictions_to_words(sample, prediction, label, label_list)
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
