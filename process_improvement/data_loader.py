import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

def list_example_files() -> list[str]:
    """
    Returns a list of CSV file names (without extension) in the data folder.
    """
    return [p.stem for p in DATA_DIR.glob("*.csv")]

def load_example_data(name: str) -> pd.DataFrame:
    """
    Load example CSV from package data folder.
    """
    file_path = DATA_DIR / f"{name}.csv"
    if not file_path.exists():
        available = list_example_files()
        raise FileNotFoundError(
            f"CSV not found: {file_path}\n"
            f"Available files: {available}"
        )
    return pd.read_csv(file_path)