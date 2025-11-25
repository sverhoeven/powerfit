# Feature Specification: CUDA Backend for GPU Acceleration

**Feature Branch**: `[001-implement-cuda-backend]`  
**Created**: 2025-11-25  
**Status**: Draft  
**Input**: User description: "$ARGUMENTS"

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.
  
  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - CUDA backend used on NVIDIA GPU (Priority: P1)

User runs `powerfit --gpu` on a workstation equipped with an NVIDIA GPU. The application automatically detects the NVIDIA hardware, selects the CUDA backend, and performs the fitting computation using CUDA kernels, resulting in faster execution.

**Why this priority**: Accelerating GPU computations on NVIDIA hardware provides the most performance benefit for powerfit users with high‑throughput workloads.

**Independent Test**: Can be tested by running `powerfit ribosome-KsgA.map 13 KsgA.pdb -a 20 --gpu -d run --delimiter , --no-progressbar`, the log should contain `Using CUDA GPU-accelerated search.`

---

### User Story 2 - OpenCL fallback on non‑NVIDIA hardware (Priority: P2)

User runs `powerfit --gpu` on a workstation without an NVIDIA GPU (e.g., AMD or Intel). The application detects the lack of NVIDIA hardware and automatically uses the OpenCL backend to perform the fitting computation.

**Why this priority**: Accelerating GPU computations on NVIDIA hardware provides the most performance benefit for powerfit users with high‑throughput workloads.

**Independent Test**: Can be tested by running `powerfit ribosome-KsgA.map 13 KsgA.pdb -a 20 -d run --delimiter , --no-progressbar`, the log should not contain `Using CUDA GPU-accelerated search.`

---

### User Story 3 - User can explicitly select backend (Priority: P3)

User runs `powerfit --gpu` on a workstation equipped with an NVIDIA GPU. The application automatically detects the NVIDIA hardware, selects the CUDA backend, and performs the fitting computation using CUDA kernels, resulting in faster execution.

**Why this priority**: Accelerating GPU computations on NVIDIA hardware provides the most performance benefit for powerfit users with high‑throughput workloads.

**Independent Test**: Can be tested by running `powerfit ribosome-KsgA.map 13 KsgA.pdb -a 20 --gpu --gpu-backend cuda -d run --delimiter , --no-progressbar`, the log should contain `Using CUDA GPU-accelerated search.`


---

[Add more user stories as needed, each with an assigned priority]

### Edge Cases

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right edge cases.
-->

- Given `--gpu --gpu-backend cuda` on a system without an NVIDIA GPU should raise a clear error message indicating that CUDA is not available.

## Requirements *(mandatory)*

<!--
  ACTION REQUIRED: The content in this section represents placeholders.
  Fill them out with the right functional requirements.
-->

### Functional Requirements

- **FR-001**: System MUST detect the presence of an NVIDIA GPU at runtime when the `--gpu` flag is used.
- **FR-002**: System MUST select the CUDA backend automatically when an NVIDIA GPU is detected and the `--gpu` flag is used.
- **FR-003**: System MUST allow users to explicitly request the CUDA backend via a command‑line option `--gpu-backend cuda` or `--gpu-backend opencl`.
- **FR-004**: System MUST fallback to the OpenCL backend when no NVIDIA GPU is detected or CUDA is unavailable.
- **FR-005**: System MUST log backend selection events, including GPU detection results and chosen backend, for diagnostic purposes.

## Success Criteria *(mandatory)*

<!--
  ACTION REQUIRED: Define measurable success criteria.
  These must be technology-agnostic and measurable.
-->

### Measurable Outcomes

- **SC-001**: The fit should be at least as fast as the existing OpenCL implementation on NVIDIA hardware.
