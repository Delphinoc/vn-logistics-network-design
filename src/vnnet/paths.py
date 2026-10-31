from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "reports" / "figures"
for p in (PROCESSED, TABLES, FIGURES):
    p.mkdir(parents=True, exist_ok=True)


def load_nodes():
    import pandas as pd
    return pd.read_csv(PROCESSED / "nodes.csv")
