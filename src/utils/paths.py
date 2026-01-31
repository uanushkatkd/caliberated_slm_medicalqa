from pathlib import Path

# repo root = folder that contains README.md
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CONFIGS_DIR = ROOT_DIR / "configs"
PROMPTS_DIR = ROOT_DIR / "prompts"
