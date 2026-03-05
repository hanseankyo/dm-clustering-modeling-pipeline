from pathlib import Path
import argparse
import runpy

MODEL_DIR = Path(__file__).resolve().parent

SCRIPTS = {
    "pretrain": MODEL_DIR / "Pre_training_code.py",
    "train": MODEL_DIR / "Main_training_code.py",
    "shap": MODEL_DIR / "SHAP.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modeling scripts")
    parser.add_argument("task", choices=sorted(SCRIPTS.keys()))
    args = parser.parse_args()

    script_path = SCRIPTS[args.task]
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
