# Quickstart for CUDA Backend

## Installation

```bash
# Install the core library
pip install powerfit-em

# Install the optional CUDA dependencies
pip install powerfit-em[cuda]
```

> **Note**: The `cuda` extra pulls in `pyvkfft` and `pycuda`. Ensure you have a working CUDA toolkit and compatible NVIDIA driver installed.

## Running PowerFit with CUDA

```bash
# Basic usage – automatically selects CUDA if available
powerfit my_target.map 10 my_template.pdb -a 20 --gpu -d results

# Force the CUDA backend (fails if no CUDA device)
powerfit my_target.map 10 my_template.pdb -a 20 --gpu --gpu-backend cuda -d results

# Fallback to OpenCL (or CPU) explicitly
powerfit my_target.map 10 my_template.pdb -a 20 --gpu --gpu-backend opencl -d results
```

The CLI will log the chosen backend, e.g.:

```
[INFO] Detected NVIDIA GPU – using CUDA backend.
```

## Minimal Example (Python API)

```python
from powerfit_em import PowerFitter, Volume
from powerfit_em.helpers import opencl_available

# Load volumes (numpy arrays)
vol_target = Volume.from_file('my_target.map')
vol_template = Volume.from_file('my_template.pdb')
vol_mask = Volume.from_file('mask.map')

# Choose backend
if opencl_available():
    # Use OpenCL if CUDA not present
    from powerfit_em.correlators.gpu import GPUCorrelator as Backend
else:
    from powerfit_em.correlators.cpu import CPUCorrelator as Backend

# Initialize fitter (queue is None for CPU/OpenCL fallback handled internally)
fit = PowerFitter(
    target=vol_target,
    rotations=vol_target.generate_rotations(),
    template=vol_template,
    mask=vol_mask,
    queue=None,
    nproc=1,
)

fit.scan(progress=None)
print('Best LCC score:', fit.lcc.max())
```

## Testing

Run the test suite to ensure the CUDA backend works (requires a CUDA‑capable CI runner):

```bash
pytest -m cuda
```

The `cuda` marker selects tests that require the optional dependencies.

---

*Prepared on 2025‑11‑25.*
