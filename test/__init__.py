import sys
from pathlib import Path

try:
    from edge import Edge
except ImportError:
    root_dir = Path(__file__).parents[1]
    sys.path.insert(0, str(root_dir))
