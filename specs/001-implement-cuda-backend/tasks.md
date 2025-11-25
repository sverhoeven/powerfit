---

description: "Task list for CUDA Backend implementation"
---

# Tasks: CUDA Backend Implementation

**Input**: Design documents from `/specs/001-implement-cuda-backend/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are included where the specification requests validation (backend detection, selection, fallback, explicit option).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

- [ ] T001 [P] Create module `src/powerfit_em/correlators/cuda.py` for the CUDA backend implementation.
- [ ] T002 Add optional dependency entry `powerfit-em[cuda]` in `pyproject.toml` (includes `pyvkfft` and `pycuda`).
- [ ] T003 Update `docs/quickstart.md` (or `specs/.../quickstart.md`) with installation and usage instructions for the CUDA extra.
- [ ] T004 Add CLI flag `--gpu-backend` with choices `cuda` and `opencl` in `src/powerfit_em/cli.py`.
- [ ] T005 Create utility `src/powerfit_em/correlators/gpu_detector.py` to detect CUDA‑capable devices.

---

## Phase 2: Foundational (Blocking Prerequisites)

- [ ] T006 Rename existing GPU correlator file `src/powerfit_em/correlators/gpu.py` to `opencl.py` and update all imports.
- [ ] T007 Ensure abstract class `Correlator` in `src/powerfit_em/correlators/shared.py` defines the full public API required by both backends.
- [ ] T008 Implement `src/powerfit_em/correlators/backend_selector.py` that chooses between `CudaCorrelator` and `OpenCLCorrelator` based on `GPUDetector` and the CLI option.

---

## Phase 3: User Story 1 - CUDA backend used on NVIDIA GPU (Priority: P1) 🎯 MVP

**Goal**: When an NVIDIA GPU is present the application automatically selects the CUDA backend.

**Independent Test**: Running `powerfit … --gpu` on a machine with CUDA should log `Using CUDA GPU‑accelerated search.` and the CUDA path should be exercised.

### Tests for User Story 1 (optional – include because spec demands verification)

- [ ] T009 [P] [US1] Unit test for `GPUDetector` in `tests/unit/test_gpu_detector.py` (mock presence of a CUDA device).
- [ ] T010 [P] [US1] Integration test `tests/integration/test_cuda_backend.py` marked with `@pytest.mark.cuda` that verifies backend selection and basic correlation works.

### Implementation for User Story 1

- [ ] T011 [US1] Implement `CudaCorrelator` class in `src/powerfit_em/correlators/cuda.py` (subclass `Correlator`).
- [ ] T012 [US1] Populate `gpu_detector.py` with detection logic using `pycuda.driver.Device.count()`.
- [ ] T013 [US1] Wire backend selection in `backend_selector.py` (or directly in CLI) to instantiate `CudaCorrelator` when `GPUDetector` reports a device.
- [ ] T014 [P] [US1] Add logging of the selected backend in `src/powerfit_em/logger.py` (or via existing logger).

**Checkpoint**: User Story 1 should be fully functional and testable independently.

---

## Phase 4: User Story 2 - OpenCL fallback on non‑NVIDIA hardware (Priority: P2)

**Goal**: When no CUDA device is present the application falls back to the existing OpenCL backend.

**Independent Test**: Running `powerfit … --gpu` on a machine without NVIDIA GPU should **not** contain the CUDA log message and should use OpenCL.

### Tests for User Story 2

- [ ] T015 [P] [US2] Unit test `tests/unit/test_opencl_fallback.py` that forces `GPUDetector` to return `False` and checks that `OpenCLCorrelator` is instantiated.
- [ ] T016 [P] [US2] Integration test `tests/integration/test_opencl_fallback.py` (no CUDA marker) that runs a small correlation job and verifies correct results.

### Implementation for User Story 2

- [ ] T017 [US2] Ensure `backend_selector.py` correctly selects `OpenCLCorrelator` when CUDA is unavailable.

**Checkpoint**: Both User Story 1 and User Story 2 can be tested independently.

---

## Phase 5: User Story 3 - Explicit backend selection (Priority: P3)

**Goal**: Users can force a specific backend via `--gpu-backend cuda|opencl`.

**Independent Test**: `powerfit … --gpu --gpu-backend cuda` on a machine without CUDA should raise a clear `RuntimeError`.

### Tests for User Story 3

- [ ] T018 [P] [US3] Unit test `tests/unit/test_cli_backend_option.py` that passes `--gpu-backend cuda` with no device and expects a `RuntimeError`.
- [ ] T019 [P] [US3] Unit test that `--gpu-backend opencl` forces the OpenCL backend even when CUDA is present.

### Implementation for User Story 3

- [ ] T020 [US3] Extend CLI parsing in `src/powerfit_em/cli.py` to validate the `--gpu-backend` argument against detector results.
- [ ] T021 [US3] Raise a descriptive error in `src/powerfit_em/cli.py` when the user requests CUDA but `GPUDetector` reports no device.

**Checkpoint**: All three user stories are independently functional and covered by tests.

---

## Phase N: Polish & Cross‑Cutting Concerns

- [ ] T022 [P] Update overall documentation (README, API docs) with the new CUDA backend description.
- [ ] T023 [P] Run quickstart validation script to ensure the updated `quickstart.md` examples execute without errors.
- [ ] T024 [P] Code cleanup: remove any stale imports of the old `gpu.py` module.
- [ ] T025 [P] Add performance benchmark script `scripts/benchmark_cuda.py` to compare CUDA vs OpenCL runtimes.

---

## Dependencies & Execution Order

- **Setup (Phase 1)**: No dependencies – can start immediately.
- **Foundational (Phase 2)**: Depends on completion of Phase 1.
- **User Stories (Phases 3‑5)**: All depend on Phase 2 but are otherwise independent; they can be worked on in parallel.
- **Polish (Final Phase)**: Depends on completion of the desired user stories.

### Parallel Opportunities

- All `[P]` tasks within a phase may be executed concurrently by different developers.
- After Phase 2, each user story phase can be assigned to a separate developer.

---

## Implementation Strategy

1. Complete **Phase 1** (Setup).
2. Complete **Phase 2** (Foundational) – this unblocks all stories.
3. Implement **User Story 1** first (MVP) and validate.
4. Proceed with **User Story 2** and **User Story 3** in any order or in parallel.
5. Finish **Polish** tasks.

---

*Generated by opencode agent.*