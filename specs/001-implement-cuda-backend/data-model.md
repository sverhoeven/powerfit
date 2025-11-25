# Data Model

## Core Entities

| Entity | Description | Fields / Attributes |
|--------|-------------|----------------------|
| **CudaCorrelator** | Subclass of `Correlator` that runs the correlation pipeline on NVIDIA GPUs using CUDA. | `vars`, `vars_ft`, `queue`, `norm_factor`, `target`, `mask`, `lcc`, `rot`, `rfftn`, `irfftn`, `conj_multiply`, `square`, `rotate_grids`, `compute_lcc_score_and_take_best`, `scan` |
| **GPUDetector** | Utility class/function that detects the presence of a CUDA‑capable device and returns a boolean flag. | `has_cuda` (bool) |
| **OpenCLCorrelator** | Existing GPU correlator (renamed from `GPUCorrelator`). | Same API as `CudaCorrelator` |

## Relationships

- `PowerFitter` (in `src/powerfit_em/powerfitter.py`) will instantiate either `CudaCorrelator` or `OpenCLCorrelator` based on the result of `GPUDetector` and the CLI flag `--gpu-backend`.
- Both correlator implementations share the abstract base class `Correlator` defined in `src/powerfit_em/correlators/shared.py`.

## Validation Rules

- The selected backend must be available; otherwise a clear `RuntimeError` is raised.
- All input arrays (`target`, `template`, `mask`) must be 3‑D float32 volumes of matching shape.
- The `target` array is normalised to its maximum value before any computation.

## State Transitions

1. **Initialisation** – Allocate GPU buffers, compile kernels, compute static FFT of the target.
2. **Template Set** – Upload template and mask, compute normalisation factor.
3. **Rotation Loop** – For each rotation: rotate grids, compute FFT‑based correlations, update LCC scores.
4. **Finalisation** – Retrieve `lcc` and `rot` arrays from the device.

---

*Prepared on 2025‑11‑25.*
