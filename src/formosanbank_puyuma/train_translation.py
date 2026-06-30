from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import load_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a translation model on Puyuma data")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="google/byt5-small")
    parser.add_argument("--source-field", type=str, default="source_text")
    parser.add_argument("--target-field", type=str, default="target_text")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/translation"))
    parser.add_argument("--run-name", type=str, default="puyuma-translation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--push-to-hub", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import numpy as np
        import evaluate
        from datasets import Dataset, DatasetDict
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit(
            "Translation training requires optional dependencies. Install with: python -m pip install .[translate]"
        ) from exc

    train_rows = load_jsonl(args.train_file)
    valid_rows = load_jsonl(args.validation_file)
    if not train_rows:
        raise SystemExit("Training set is empty")
    if not valid_rows:
        raise SystemExit("Validation set is empty")

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(valid_rows),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        model_inputs = tokenizer(
            [str(item) for item in batch[args.source_field]],
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=[str(item) for item in batch[args.target_field]],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=args.num_workers if args.num_workers > 1 else None,
    )

    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_predictions = [item.strip() for item in decoded_predictions]
        decoded_labels = [[item.strip()] for item in decoded_labels]
        result = bleu.compute(predictions=decoded_predictions, references=decoded_labels)
        result["gen_len"] = float(np.mean([len(item.split()) for item in decoded_predictions])) if decoded_predictions else 0.0
        return {key: round(value, 4) if isinstance(value, float) else value for key, value in result.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir / args.run_name),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        report_to=[],
        push_to_hub=args.push_to_hub,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(args.output_dir / args.run_name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
