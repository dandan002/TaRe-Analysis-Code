import sys
from pathlib import Path

# Ensure tare_analysis/ root is on sys.path so bare imports like
# `from config import ...` and `from xps import ...` always resolve,
# regardless of which directory pytest is invoked from.
sys.path.insert(0, str(Path(__file__).parent))
