"""Run the full pipeline in order (about 1 minute on a laptop)."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for stage in ["01_build_nodes.py", "02_hub_sweep.py", "03_scenarios.py", "04_sensitivity.py", "05_figures.py"]:
    print(f"\n=== {stage} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / stage)], check=True, cwd=HERE)
