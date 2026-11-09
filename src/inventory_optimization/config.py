from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "synthetic_context_aware_inventory_dataset.csv"
DATA_DICTIONARY = ROOT / "data" / "raw" / "inventory_dataset_data_dictionary.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 28)
