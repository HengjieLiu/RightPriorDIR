"""Small adapters to pinned upstream source, with no global networks-name clash."""

from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_EXTERNAL_COMMIT = "f7c8e833435ec9c6b7ebec6c9a8098b566709146"


def verify(name):
    """Content verification also works in Docker without Git metadata."""
    entry = json.loads((ROOT / "third_party/lock.json").read_text())[name]
    repo = ROOT / "third_party" / name
    for relative, expected in entry["files"].items():
        path = repo / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError(
                f"{name} content mismatch: {relative}; restore the pinned submodule"
            )
    return repo, entry["commit"]


@lru_cache(None)
def network_module(name):
    if name not in {"IDIR", "SINR"}:
        raise ValueError(name)
    repo, _ = verify(name)
    spec = importlib.util.spec_from_file_location(
        f"rightpriordir_upstream_{name}", repo / "networks/networks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def siren(name, layers, omega):
    return network_module(name).Siren(layers, weight_init=True, omega=omega)


@lru_cache(None)
def load_external_api():
    repo, commit = verify("DUAL-INR-DIR")
    source = repo / "src"
    existing = sys.modules.get("inrdir")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(
        source.resolve()
    ):
        raise RuntimeError("An unrelated inrdir installation was already imported")
    sys.path.insert(0, str(source))
    try:
        modules = {
            name: importlib.import_module(f"inrdir.{name}")
            for name in (
                "configs",
                "model",
                "ncc",
                "regularizers",
                "scheduler",
                "trainer",
                "utils",
            )
        }
    finally:
        sys.path.remove(str(source))
    return SimpleNamespace(
        commit=commit, repository=repo, package_dir=source / "inrdir", **modules
    )
