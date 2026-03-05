from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
SNP_FEATURE_NUM = 31

# Shared model constants
PRE_FEATURE_NUM = 18

# Pretraining defaults
PRETRAIN_BATCH_SIZE = 512
PRETRAIN_EPOCHS = 5000
PRETRAIN_EARLY = 5000
PRETRAIN_LR = 1e-2

# Transfer learning defaults
TRANSFER_EARLY = 30
TRANSFER_WARMUP_EPOCHS = 10
TRANSFER_TOTAL_EPOCHS = 7000
TRANSFER_BATCH_SIZE_LIST = [1024]
TRANSFER_RANDOM_STATE_LIST = [123]
TRANSFER_LEARNING_RATE_LIST = [1e-2]
TRANSFER_SNP_NUM_LIST = [3]
TRANSFER_TOTAL_NUM_LIST = [3]
TRANSFER_POSITIVE_WEIGHT_LIST = [1.7]
TRANSFER_DROPOUT_NUM_LIST = [0.5]

# SHAP evaluation default
SHAP_DEFAULT_POS_WEIGHT = 8.0

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
