"""Check frozen inputs, pinned upstream contents and runtime without training."""

import argparse
import importlib.metadata
import json
from .frozen import ROOT, verify_inputs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", action="store_true")
    a = p.parse_args(argv)
    manifest = verify_inputs()
    from .upstream import verify, load_external_api, siren
    from .register import resolve_config
    import torch

    result = {
        "frozen_inputs": len(manifest["inputs"]),
        "submodules": {},
        "configurations": 0,
        "optimization_run": False,
    }
    for name in ["IDIR", "SINR", "DUAL-INR-DIR"]:
        result["submodules"][name] = verify(name)[1]
    for path in (ROOT / "configs").rglob("*.json"):
        resolve_config(path)
        result["configurations"] += 1
    from .experiment import SUITES, build_plan

    result["experiment_suites"] = {
        name: len(build_plan(name)["jobs"]) for name in SUITES
    }
    for name in ["IDIR", "SINR"]:
        siren(name, [3, 8, 3], 32)
    load_external_api()
    result["versions"] = {
        name: importlib.metadata.version(name)
        for name in [
            "torch",
            "numpy",
            "scipy",
            "matplotlib",
            "nibabel",
            "pandas",
            "SimpleITK",
            "surface-distance",
            "monai",
            "lungmask",
        ]
    }
    if a.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU not visible; check host driver and NVIDIA Container Toolkit"
            )
        result["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
        value = torch.ones(1, device="cuda") + 1
        assert value.item() == 2
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
