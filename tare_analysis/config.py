# tare_analysis/config.py
from pathlib import Path
from datetime import datetime

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
QUBITS_DATA_DIR = ROOT.parent / "XRD_analysis" / "data"

def make_run_dirs(run_suffix: str = "") -> tuple[Path, Path]:
    """Create timestamped results/YYYY-MM-DD_HH-MM-SS[_SUFFIX]/{figures,csv} and return (figures_dir, csv_dir)."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = stamp if not run_suffix else f"{stamp}_{run_suffix}"
    run   = ROOT / "results" / run_name
    figs  = run / "figures"
    csvs  = run / "csv"
    for d in [figs, csvs]:
        d.mkdir(parents=True, exist_ok=True)
    return figs, csvs
