import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parent

for path in (REPO_DIR, PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
