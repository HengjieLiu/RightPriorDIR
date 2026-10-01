"""All release acceptance tests prohibit real optimizer steps."""

import pytest
import torch


@pytest.fixture(autouse=True)
def no_optimizer_steps(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Real optimization is outside release acceptance")

    monkeypatch.setattr(torch.optim.Adam, "step", fail)
    monkeypatch.setattr(torch.optim.AdamW, "step", fail)
