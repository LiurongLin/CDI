from __future__ import annotations

from . import cdi_reports as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})
