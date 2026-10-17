import importlib.util
import os
from pathlib import Path

_root = Path(os.environ.get("HOMMI_ROOT", Path(__file__).resolve().parents[3] / "third_party" / "hommi"))
_spec = importlib.util.spec_from_file_location(
    "_hommi_vendored_p3d", _root / "hommi/train_network/model/vision_3d/pytorch3d_transforms.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
