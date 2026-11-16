"""Paper baselines with a bundled common result/data contract."""
import sys
from pathlib import Path

# Existing algorithms import ``baselines`` and ``common``. Both resolve inside
# this directory when the artifact is copied to another host.
_package_root = Path(__file__).resolve().parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))
sys.modules.setdefault("baselines", sys.modules[__name__])

