from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PhaseMask(ABC):
    """Base class for focal-plane phase masks."""

    @abstractmethod
    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return complex transmission map on focal-plane coordinates."""


class FlatPhaseMask(PhaseMask):
    """Constant phase mask: exp(i * phase_rad), defaulting to zero phase shift."""

    def __init__(self, phase_rad: float = 0.0):
        self.phase_rad = float(phase_rad)

    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(1j * self.phase_rad) * np.ones_like(x, dtype=np.complex128)


class NoPhaseMask(PhaseMask):
    """Backward-compatible alias for a zero-phase flat mask."""

    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.ones_like(x, dtype=np.complex128)


class VortexPhaseMask(PhaseMask):
    """Vortex phase mask: exp(i * charge * theta)."""

    def __init__(self, charge: int = 2):
        self.charge = charge

    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        theta = np.arctan2(y, x)
        return np.exp(1j * self.charge * theta)


class FQPMPhaseMask(PhaseMask):
    """Four-Quadrant Phase Mask (pi phase shift in alternating quadrants)."""

    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        q = np.sign(x) * np.sign(y)
        phase = np.where(q >= 0, 0.0, np.pi)
        return np.exp(1j * phase)


class RoddierPhaseMask(PhaseMask):
    """Roddier phase mask: central disk with a phase shift, zero phase outside."""

    def __init__(self, radius_lamD: float = 0.53, phase_rad: float = np.pi):
        self.radius_lamD = float(radius_lamD)
        self.phase_rad = float(phase_rad)

    def transmission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.sqrt(x**2 + y**2)
        phase = np.where(r <= self.radius_lamD, self.phase_rad, 0.0)
        return np.exp(1j * phase)
