"""Pytest configuration and shared fixtures for all test suites.

Pre-mocks heavy dependencies (HuggingFace AutoModel/AutoConfig, torchvision)
that cause import errors in environments without full GPU driver support.
These mocks are installed into sys.modules before any test module is collected,
so the actual model weights are never downloaded in unit/integration tests.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _stub_module(name: str) -> MagicMock:
    """Create a MagicMock that behaves like an importable module."""
    mod = MagicMock(spec=types.ModuleType(name))
    mod.__name__ = name
    mod.__package__ = name.rsplit(".", 1)[0] if "." in name else name
    return mod


# ── Stub torchvision to avoid the nms kernel registration error ───────────────
# The error "operator torchvision::nms does not exist" is raised at import time
# when transformers tries to import from torchvision.transforms.
_torchvision_stub = _stub_module("torchvision")
_torchvision_stub.transforms = _stub_module("torchvision.transforms")
_torchvision_stub.transforms.InterpolationMode = MagicMock()
sys.modules.setdefault("torchvision", _torchvision_stub)
sys.modules.setdefault("torchvision.transforms", _torchvision_stub.transforms)
sys.modules.setdefault("torchvision.datasets", _stub_module("torchvision.datasets"))
sys.modules.setdefault("torchvision.io", _stub_module("torchvision.io"))
sys.modules.setdefault("torchvision.models", _stub_module("torchvision.models"))
sys.modules.setdefault("torchvision.ops", _stub_module("torchvision.ops"))
sys.modules.setdefault("torchvision.utils", _stub_module("torchvision.utils"))
sys.modules.setdefault("torchvision._meta_registrations", _stub_module("torchvision._meta_registrations"))
