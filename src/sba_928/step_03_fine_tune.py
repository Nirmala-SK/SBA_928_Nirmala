"""Step 3: supervised fine-tuning of a seq2seq model. Model parameters change here."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sba_928.io_utils import write_json
from sba_928.modeling import resolve_device
from sba_928.prompts import format_training_input
from sba_928.settings import TrainConfig

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Train on the train split, select the best checkpoint on validation only."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.json")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    config = TrainConfig.from_json(args.config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only train and validation are loaded here. The test file is never read.
    raw = load_dataset(
        "json",
        data_files={"train": config.train_file, "validation": config.validation_file},
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        sources = [
            format_training_input(i, c)
            for i, c in zip(batch["instruction"], batch["context"], strict=True)
        ]
        model_inputs = tokenizer(
            sources, max_length=config.max_source_length, truncation=True
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=config.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)

    args_out = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="epoch",
        seed=config.seed,
        use_cpu=device == "cpu",
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args_out,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    final_metrics = trainer.evaluate()

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    write_json(
        out_dir / "training_config_resolved.json",
        {**config.to_dict(), "resolved_device": device},
    )
    write_json(
        out_dir / "training_summary.json",
        {
            "base_model": config.model_name,
            "device": device,
            "train_records": len(tokenized["train"]),
            "validation_records": len(tokenized["validation"]),
            "train_metrics": train_result.metrics,
            "final_validation_metrics": final_metrics,
            "log_history": trainer.state.log_history,
            "test_split_used": False,
        },
    )
    LOGGER.info("Saved fine-tuned model to %s", final_dir)


if __name__ == "__main__":
    main()