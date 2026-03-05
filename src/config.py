from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
SNP_FEATURE_NUM = 31

# Ensure common output dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def resolve_input(filename: str) -> Path:
    """Resolve input file from data/ first, then outputs/."""
    data_path = DATA_DIR / filename
    if data_path.exists():
        return data_path
    output_path = OUTPUT_DIR / filename
    if output_path.exists():
        return output_path
    return data_path
