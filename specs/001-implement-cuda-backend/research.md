# Research Findings for CUDA Backend Implementation

## Technical Decisions

| Decision | Rationale | Alternatives Considered |
|----------|-----------|--------------------------|
| **Use `pyvkfft` in CUDA mode** | Provides high‑performance FFTs on NVIDIA GPUs and integrates with `pycuda` buffers, matching the existing OpenCL FFT usage. | `cupy.fft` (requires full CuPy stack), custom CUDA kernels (more development effort). |
| **Use `pycuda` for data structures** | `pycuda` offers Pythonic wrappers for CUDA memory buffers, easy interoperability with `pyvkfft`, and straightforward kernel compilation. | `numba.cuda` (less direct control over kernel source), `cupy` (different API). |
| **Use `pycuda` (or `pycuda.driver` utilities) for host↔GPU data movement** | `pycuda` abstracts memory copies and synchronisation, simplifying the port from OpenCL. | Manual `cuda.memcpy_htod`/`dtoc` calls (more boilerplate). |
| **Maintain existing OpenCL implementation unchanged** | Allows fallback to OpenCL when CUDA is unavailable, satisfying the requirement for graceful degradation. | Remove OpenCL path (breaks existing users). |

## Implementation Approach

1. **Kernel Porting**: Translate the OpenCL kernel source located in `src/powerfit_em/correlators/kernels.cl` and the helper functions in `src/powerfit_em/correlators/clkernels.py` to CUDA C kernels. Keep the kernel signatures compatible with the existing `GPUCorrelator` logic.
2. **Class Refactor**:
   * Rename current `GPUCorrelator` to `OpenCLCorrelator`.
   * Introduce new abstract base class `Correlator` (already present) defining the common API (`prepare`, `run`, `finalize`).
   * Implement `CudaCorrelator(Correlator)` that loads the compiled CUDA kernels via `pycuda.compiler.SourceModule`, manages `pycuda` buffers, and invokes `pyvkfft` for FFT steps.
3. **Backend Selection**:
   * Add a `GPUDetector` utility that queries CUDA devices via `pycuda.driver.Device.count()` and checks vendor strings.
   * In the CLI entry point, after parsing `--gpu`, call the detector; if an NVIDIA device is present, instantiate `CudaCorrelator`, otherwise fall back to `OpenCLCorrelator` (or CPU). Log the selected backend.
4. **Testing Strategy**:
   * Unit tests for `GPUDetector` mocking device presence.
   * Integration tests that run a small benchmark on CI with a CPU‑only runner, ensuring the fallback path works.
   * Optional GPU‑enabled CI job (if GPU runners are available) to verify performance gain.

## Compatibility & Performance Goals

* **Functional parity**: The CUDA backend must produce results bit‑compatible with the OpenCL and CPU backends for the same input data. By comparing generated `solutions.out` CSV file.
* **Performance**: Target similar speed or better on an NVIDIA RTX 3080 compared to the OpenCL implementation (measured on the provided benchmark dataset).
* **Memory footprint**: Adding CUDA kernels should not increase the overall package size by more than 10 MB for users without CUDA support.

## Open Questions Resolved

* **Python version** – The project uses Python 3.11; all selected libraries support this version.
* **Target platform** – Primarily Linux x86_64 with NVIDIA driver ≥450.0; Windows support can be added later.
* **Testing framework** – Existing `pytest` setup will be extended; no additional test framework required.
* **License compatibility** – `pyvkfft` and `pycuda` are BSD‑licensed, compatible with the project's MIT license.

---

*Prepared by the planning agent on 2025‑11‑25.*
