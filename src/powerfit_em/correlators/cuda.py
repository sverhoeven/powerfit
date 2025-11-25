# src/powerfit_em/correlators/cuda.py
"""CUDA backend stub for powerfit.

This module provides a placeholder implementation of the CUDA correlator.  The
real CUDA backend would use *pycuda* (and possibly *pyvkfft*) to perform the same
operations as the OpenCL backend.  Implementing the full kernel conversion is a
large task; for now we expose a class with the same public API that raises a
clear error if instantiated.  This allows the rest of the codebase to import the
module without breaking, and future work can replace the stub with a functional
implementation.
"""

from __future__ import annotations

from typing import Any

# The shared abstract base class defines the required public methods.
# Importing it does not require any GPU‑specific libraries.
from powerfit_em.correlators.shared import Correlator


class CudaCorrelator(Correlator):
    """Placeholder for the CUDA backend.

    The constructor mirrors the signature of the OpenCL correlator so that the
    backend selector can instantiate it transparently.  As soon as the class is
    used it raises ``RuntimeError`` with a helpful message.
    """

    def __init__(
        self,
        target: Any,
        template: Any,
        rotations: Any,
        mask: Any,
        queue: Any | None = None,
        laplace: bool = False,
    ) -> None:
        # ``queue`` is kept for API compatibility – a CUDA implementation would
        # receive a ``pycuda.driver.Context`` or similar object.
        raise RuntimeError(
            "CUDA backend not yet implemented. Install the optional "
            "'cuda' extra and provide a concrete CudaCorrelator implementation."
        )

    # The following methods are defined only to satisfy the abstract base class.
    # They simply raise the same error to make the failure mode obvious.
    def _set_template_var(self, template: Any) -> None:  # pragma: no cover
        raise RuntimeError("CUDA backend not implemented.")

    def _set_mask_var(self, mask: Any) -> None:  # pragma: no cover
        raise RuntimeError("CUDA backend not implemented.")

    def rotate_grids(self, rotmat: Any) -> None:  # pragma: no cover
        raise RuntimeError("CUDA backend not implemented.")

    def compute_lcc_score_and_take_best(self, n: int) -> None:  # pragma: no cover
        raise RuntimeError("CUDA backend not implemented.")

    def retrieve_results(self) -> None:  # pragma: no cover
        raise RuntimeError("CUDA backend not implemented.")

    def scan(self, progress: Any | None = None) -> None:  # pragma: no cover
        raise RuntimeError("CUDA backend not implemented.")

    # The base class also provides ``set_template`` which calls the private
    # helpers above; we inherit that behaviour unchanged.


"""End of CUDA backend stub."""
