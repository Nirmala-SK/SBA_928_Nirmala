"""Project paths and the training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from sba_928.io_utils import read_json

SEED = 42
BASE_MODEL = "google/flan-t5-small"

RAW_CSV = Path("data/raw/synthetic_market_reviews.csv")
TRAIN_FILE = Path("data/processed/train.jsonl")
VALIDATION_FILE = Path("data/processed/validation.jsonl")
TEST_FILE = Path("data/processed/test.jsonl")
MANIFEST_FILE = Path("data/processed/manifest.json")

PROMPT_DIR = Path("artifacts/prompt_baseline")
EVAL_DIR = Path("artifacts/evaluation")
BIAS_DIR = Path("artifacts/bias_audit")
DEFAULT_FINE_TUNED = Path("artifacts/models/flan-t5-sba-928/final")


@dataclass
class TrainConfig:
    """Hyperparameters and paths for fine-tuning."""

    model_name: str = BASE_MODEL
    train_file: str = str(TRAIN_FILE)
    validation_file: str = str(VALIDATION_FILE)
    test_file: str = str(TEST_FILE)
    output_dir: str = "artifacts/models/flan-t5-sba-928"
    max_source_length: int = 256
    max_target_length: int = 96
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    seed: int = SEED
    device: str = "auto"

    @classmethod
    def from_json(cls, path: str | Path) -> TrainConfig:
        """Load a config file; unknown keys raise so typos are caught."""
        return cls(**read_json(path))

    def to_dict(self) -> dict[str, object]:
        """Return the config as a plain dictionary."""
        return asdict(self)