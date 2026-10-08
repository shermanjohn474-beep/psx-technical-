import os
os.environ.setdefault("PSX_SKIP_DOTENV", "1")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
