# Implementation Plan: [FEATURE]

**Branch**: `[001-implement-cuda-backend]` | **Date**: 2025-11-25 | **Spec**: `/specs/001-implement-cuda-backend/spec.md`
**Input**: Feature specification from `/specs/[###-feature-name]/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a CUDA backend for the correlation engine in `powerfit-em`. The backend will use **pyvkfft** for FFT operations (leveraging its CUDA support) and **pycuda** to run the existing OpenCL kernels (converted to CUDA) on NVIDIA GPUs. A new `CudaCorrelator` class will subclass the existing `Correlator` base in `src/powerfit_em/correlators/shared.py`. The implementation must automatically fall back to the OpenCL backend when CUDA is unavailable, and expose the backend selection via the `--gpu-backend` CLI option.

Key steps include:
- Research pyvkfft CUDA usage and pycuda data handling.
- Design the `CudaCorrelator` API matching the existing correlator interface.
- Port `kernels.cl` to CUDA kernels (or use pycuda ElementwiseKernel where possible).
- Implement data movement between host and GPU using pycuda GPUArray.
- Add optional dependency entry `powerfit-em[cuda]` in `pyproject.toml`.
- Update documentation and quick‑start examples.
- Ensure comprehensive pytest coverage and performance benchmarks.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11  
**Primary Dependencies**: pyvkfft, pycuda (or cupy), numpy, powerfit-em (existing)  
**Storage**: N/A (in‑memory GPU buffers)  
**Testing**: pytest, pytest‑cov  
**Target Platform**: Linux (CUDA‑capable NVIDIA GPU)  
**Project Type**: single library (powerfit-em)  
**Performance Goals**: CUDA implementation should be ≥1.5× faster than current OpenCL backend on comparable hardware  
**Constraints**: Must fallback to OpenCL backend when CUDA not available; keep memory usage ≤2× current OpenCL version  
**Scale/Scope**: Support typical protein‑fit workloads (arrays up to ~200 × 200 × 200 voxels)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

[Gates determined based on constitution file]

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
src/powerfit_em/
├── __init__.py
├── _extensions.c
├── _powerfit.pyx
├── analyzer.py
├── elements.py
├── helpers.py
├── powerfit.py
├── powerfitter.py
├── report.py
├── rotations.py
├── shape_descriptor.py
├── structure.py
├── volume.py
├── data/
│   ├── E.npy
│   ├── README
│   └── *.npy (sample density maps)
└── correlators/
    ├── __init__.py
    ├── clkernels.py
    ├── cpu.py
    ├── gpu.py
    ├── kernels.cl
    ├── shared.py
    └── (future) cuda.py (CUDA backend implementation)
```

**Structure Decision**: The code resides under `src/powerfit_em/correlators/` with a new `cuda.py` module for the CUDA backend. Existing OpenCL implementation stays in `gpu.py` (renamed to `opencl.py`). The correlator hierarchy is:
```
Correlator (shared)
├── OpenCLCorrelator (gpu.py renamed)
└── CudaCorrelator (cuda.py new)
```

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
