# Performance measurement report

[View measured source](https://github.com/Qiaoyi-Li/FiniteMPS.jl/commit/a19056f4b950167a7207ace2ca4c845c2d644393) · [Download raw data (JSON)](report.json) · [View workflow run](https://github.com/Qiaoyi-Li/FiniteMPS.jl/actions/runs/42017838766)

## Measurement environment and thread settings

| Field | Recorded at measurement time |
| --- | --- |
| Repository | Qiaoyi-Li/FiniteMPS.jl |
| Source commit | a19056f4b950167a7207ace2ca4c845c2d644393 |
| Benchmark definition commit | a19056f4b950167a7207ace2ca4c845c2d644393 |
| Uncommitted changes | No |
| Version tag | Not recorded |
| Release type | Development or local build |
| Algorithm library version | 2.0.0 |
| Measured at (UTC) | 2026-11-05T14:52:17.159Z |
| Processor model | AMD EPYC 7763 64-Core Processor |
| Processor architecture | 64-bit x86 |
| Visible logical processors | 4 |
| Processors available to this process | 4 |
| Allowed processor identifiers | 0-3 |
| Visible physical cores | 2 |
| Julia version | 1.11.6 |
| Julia computation threads | 4 |
| Julia interactive threads | 0 |
| Garbage collection threads | 1 |
| Matrix computation threads | 1 |
| Matrix computation backend | OpenBLAS |
| Runner label | ubuntu-24.04 |
| Runner type | GitHub-hosted runner |
| Runner image version | 20261026.354.1 |
| Operating system | Linux |
| System kernel | 7.0.0-1017-azure |
| Visible system memory (bytes) | 16760344576 |
| Run trigger | Commit push |
| Automation workflow | Performance |
| Run identifier | 42017838766 |
| Workflow run number | 6 |
| Run attempt | 1 |
| Benchmark definition file | benchmark/benchmarks.jl |
| Benchmark definition source | Current source checkout |

Processor counts and thread settings describe available resources, not runtime core utilization. System memory is the visible capacity; allocated memory is the amount allocated by the measured operation. Neither value is peak process memory.

## Measurements

| Operation and size | Median time | Total allocated bytes | Memory allocation count | Samples |
| --- | ---: | ---: | ---: | ---: |
| Hubbard / U1SU2 · 2-DMRG · D=128 | 10.287 seconds | 17122519176 | 151654303 | 1 |
| Hubbard / U1SU2 · 2-DMRG · D=256 | 12.16 seconds | 20048504520 | 170961038 | 1 |
| Hubbard / U1SU2 · 2-DMRG · D=512 | 16.666 seconds | 25565497936 | 199879287 | 1 |
| Hubbard / U1SU2 · 2-TDVP · D=128 | 103.62 seconds | 160530135016 | 1269717660 | 1 |
| Hubbard / U1SU2 · 2-TDVP · D=256 | 312.88 seconds | 495390378960 | 3688612780 | 1 |
| Hubbard / U1SU2 · 2-TDVP · D=512 | 422.74 seconds | 606846669512 | 4307711575 | 1 |
| Hubbard / U1SU2 · CBE-DMRG · D=1024 | 17.987 seconds | 21033365016 | 102447705 | 1 |
| Hubbard / U1SU2 · CBE-DMRG · D=256 | 5.8552 seconds | 9872305904 | 92638843 | 1 |
| Hubbard / U1SU2 · CBE-DMRG · D=512 | 8.6651 seconds | 13348617272 | 110118071 | 1 |
| Hubbard / U1SU2 · CBE-TDVP · D=1024 | 62.69 seconds | 84303227968 | 528651859 | 1 |
| Hubbard / U1SU2 · CBE-TDVP · D=256 | 25.737 seconds | 42159086968 | 345177542 | 1 |
| Hubbard / U1SU2 · CBE-TDVP · D=512 | 32.056 seconds | 51530333192 | 386518801 | 1 |
| Hubbard / U1U1 · 2-DMRG · D=128 | 6.526 seconds | 8559010792 | 90763291 | 1 |
| Hubbard / U1U1 · 2-DMRG · D=256 | 10.739 seconds | 13253370888 | 89031449 | 1 |
| Hubbard / U1U1 · 2-DMRG · D=512 | 31.222 seconds | 31067886624 | 99638727 | 1 |
| Hubbard / U1U1 · 2-TDVP · D=128 | 90.273 seconds | 123035037128 | 995798379 | 1 |
| Hubbard / U1U1 · 2-TDVP · D=256 | 176.8 seconds | 229991368264 | 1407376472 | 1 |
| Hubbard / U1U1 · 2-TDVP · D=512 | 452.54 seconds | 486120204336 | 1585800968 | 1 |
| Hubbard / U1U1 · CBE-DMRG · D=1024 | 86.297 seconds | 65421735136 | 34306297 | 1 |
| Hubbard / U1U1 · CBE-DMRG · D=256 | 5.4826 seconds | 6383284568 | 34768979 | 1 |
| Hubbard / U1U1 · CBE-DMRG · D=512 | 16.03 seconds | 17250245424 | 37514426 | 1 |
| Hubbard / U1U1 · CBE-TDVP · D=1024 | 171.27 seconds | 138037916744 | 141761971 | 1 |
| Hubbard / U1U1 · CBE-TDVP · D=256 | 13.85 seconds | 17976557192 | 112339801 | 1 |
| Hubbard / U1U1 · CBE-TDVP · D=512 | 39.594 seconds | 43583741584 | 123097087 | 1 |
| Hubbard / Z2SU2 · 2-DMRG · D=128 | 2.1823 seconds | 2058614088 | 18907775 | 1 |
| Hubbard / Z2SU2 · 2-DMRG · D=256 | 4.0812 seconds | 3920480360 | 22372211 | 1 |
| Hubbard / Z2SU2 · 2-DMRG · D=512 | 12.377 seconds | 10145335584 | 29181935 | 1 |
| Hubbard / Z2SU2 · 2-TDVP · D=128 | 21.744 seconds | 23822135616 | 119268058 | 1 |
| Hubbard / Z2SU2 · 2-TDVP · D=256 | 77.902 seconds | 76612167264 | 281449073 | 1 |
| Hubbard / Z2SU2 · 2-TDVP · D=512 | 262.72 seconds | 173978917824 | 290795131 | 1 |
| Hubbard / Z2SU2 · CBE-DMRG · D=1024 | 43.417 seconds | 25863632840 | 11422701 | 1 |
| Hubbard / Z2SU2 · CBE-DMRG · D=256 | 2.6009 seconds | 2592421312 | 11572951 | 1 |
| Hubbard / Z2SU2 · CBE-DMRG · D=512 | 8.2318 seconds | 7071595824 | 13109536 | 1 |
| Hubbard / Z2SU2 · CBE-TDVP · D=1024 | 133.11 seconds | 73798108472 | 57739171 | 1 |
| Hubbard / Z2SU2 · CBE-TDVP · D=256 | 7.7435 seconds | 8099353584 | 29106819 | 1 |
| Hubbard / Z2SU2 · CBE-TDVP · D=512 | 26.868 seconds | 22390589568 | 41135278 | 1 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=128 | 6.945 seconds | 9531133416 | 89381520 | 1 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=256 | 7.9759 seconds | 10908999848 | 97266528 | 1 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=512 | 10.491 seconds | 14526705088 | 114282055 | 1 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=128 | 78.719 seconds | 123162602088 | 1020242260 | 1 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=256 | 95.076 seconds | 148542509448 | 1192434498 | 1 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=512 | 148.04 seconds | 219842303672 | 1621562164 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=1024 | 13.525 seconds | 16544056040 | 68112279 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=256 | 5.3525 seconds | 6874898776 | 63409684 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=512 | 6.769 seconds | 9650646208 | 73271861 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=1024 | 48.167 seconds | 64584525464 | 408892729 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=256 | 15.917 seconds | 25654010080 | 213589844 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=512 | 21.688 seconds | 34074614968 | 245517764 | 1 |

### Hubbard / U1SU2 · 2-DMRG · D=128

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 126, 126, 128, 128, 128, 127, 125, 128, 126, 128, 127, 128, 128, 128, 128, 128, 128, 126, 125, 127, 128, 128, 127, 127, 126, 64, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D128.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · 2-DMRG · D=256

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 255, 255, 255, 256, 254, 256, 256, 255, 253, 253, 256, 256, 256, 256, 254, 254, 254, 254, 255, 255, 253, 256, 256, 256, 253, 64, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D256.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · 2-DMRG · D=512

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 511, 511, 512, 511, 508, 510, 510, 511, 512, 512, 511, 511, 512, 512, 512, 511, 510, 509, 512, 512, 508, 512, 512, 256, 64, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D512.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · 2-TDVP · D=128

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 127, 128, 127, 128, 125, 128, 126, 128, 127, 126, 126, 126, 127, 128, 126, 128, 127, 128, 126, 128, 127, 127, 126, 126, 126, 128, 128, 128, 127, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D128.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · 2-TDVP · D=256

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 255, 256, 254, 256, 255, 255, 256, 254, 253, 255, 256, 256, 255, 256, 255, 254, 254, 253, 255, 256, 256, 256, 254, 256, 256, 255, 255, 256, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D256.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · 2-TDVP · D=512

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 510, 512, 511, 512, 511, 512, 509, 510, 512, 512, 512, 512, 509, 512, 510, 512, 511, 512, 511, 512, 509, 512, 509, 510, 512, 511, 512, 256, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D512.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · CBE-DMRG · D=1024

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 1016, 1023, 1024, 1023, 1024, 1022, 1020, 1022, 1024, 1024, 1023, 1022, 1022, 1024, 1023, 1020, 1024, 1023, 1023, 1022, 1024, 1024, 1015, 256, 64, 16, 4, 1 |
| cbe_target | 2048 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 1024 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D1024.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · CBE-DMRG · D=256

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 255, 255, 255, 256, 254, 256, 256, 255, 253, 253, 256, 256, 256, 256, 254, 254, 254, 254, 255, 255, 253, 256, 256, 256, 253, 64, 16, 4, 1 |
| cbe_target | 512 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D256.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · CBE-DMRG · D=512

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 511, 511, 512, 511, 508, 510, 510, 511, 512, 512, 511, 511, 512, 512, 512, 511, 510, 509, 512, 512, 508, 512, 512, 256, 64, 16, 4, 1 |
| cbe_target | 1024 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D512.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · CBE-TDVP · D=1024

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 1022, 1024, 1024, 1024, 1023, 1022, 1022, 1023, 1022, 1022, 1020, 1022, 1023, 1024, 1024, 1024, 1024, 1021, 1022, 1024, 1024, 1023, 1024, 1023, 1023, 1023, 1022, 256, 16, 1 |
| cbe_target | 1152 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 1024 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D1024.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · CBE-TDVP · D=256

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 255, 256, 254, 256, 255, 255, 256, 254, 253, 255, 256, 256, 255, 256, 255, 254, 254, 253, 255, 256, 256, 256, 254, 256, 256, 255, 255, 256, 16, 1 |
| cbe_target | 288 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D256.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1SU2 · CBE-TDVP · D=512

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 510, 512, 511, 512, 511, 512, 509, 510, 512, 512, 512, 512, 509, 512, 510, 512, 511, 512, 511, 512, 509, 512, 509, 510, 512, 511, 512, 256, 16, 1 |
| cbe_target | 576 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D512.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · 2-DMRG · D=128

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D128.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · 2-DMRG · D=256

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 255, 64, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D256.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · 2-DMRG · D=512

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 64, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D512.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · 2-TDVP · D=128

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D128.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · 2-TDVP · D=256

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D256.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · 2-TDVP · D=512

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D512.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · CBE-DMRG · D=1024

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 1017, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1017, 256, 64, 16, 4, 1 |
| cbe_target | 2048 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 1024 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D1024.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · CBE-DMRG · D=256

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 255, 64, 16, 4, 1 |
| cbe_target | 512 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D256.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · CBE-DMRG · D=512

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 64, 16, 4, 1 |
| cbe_target | 1024 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D512.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · CBE-TDVP · D=1024

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 256, 16, 1 |
| cbe_target | 1152 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 1024 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D1024.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · CBE-TDVP · D=256

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 16, 1 |
| cbe_target | 288 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D256.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / U1U1 · CBE-TDVP · D=512

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 16, 1 |
| cbe_target | 576 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D512.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · 2-DMRG · D=128

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 128, 128, 128, 128, 127, 128, 128, 128, 127, 128, 128, 128, 127, 128, 128, 128, 127, 128, 128, 128, 127, 128, 128, 128, 128, 64, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D128.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · 2-DMRG · D=256

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 253, 255, 254, 256, 256, 256, 256, 255, 253, 256, 256, 256, 253, 256, 256, 255, 256, 256, 256, 256, 254, 256, 253, 256, 64, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D256.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · 2-DMRG · D=512

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 510, 509, 509, 512, 510, 510, 511, 510, 510, 510, 512, 510, 511, 510, 510, 510, 512, 508, 511, 510, 511, 508, 511, 256, 64, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D512.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · 2-TDVP · D=128

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 127, 128, 127, 128, 128, 126, 126, 126, 127, 126, 128, 126, 127, 126, 126, 126, 127, 126, 126, 126, 125, 128, 126, 126, 127, 128, 128, 128, 127, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D128.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · 2-TDVP · D=256

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 255, 255, 254, 256, 255, 255, 256, 256, 256, 255, 255, 256, 255, 255, 254, 256, 256, 256, 256, 254, 255, 253, 253, 254, 254, 255, 256, 256, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D256.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · 2-TDVP · D=512

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 510, 510, 512, 510, 509, 512, 511, 512, 510, 512, 510, 510, 511, 512, 511, 512, 510, 512, 510, 512, 511, 512, 509, 510, 510, 511, 512, 256, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D512.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · CBE-DMRG · D=1024

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 1024, 1023, 1024, 1023, 1023, 1022, 1024, 1023, 1022, 1022, 1022, 1022, 1020, 1022, 1023, 1023, 1022, 1022, 1020, 1023, 1024, 1023, 1024, 256, 64, 16, 4, 1 |
| cbe_target | 2048 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 1024 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D1024.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · CBE-DMRG · D=256

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 253, 255, 254, 256, 256, 256, 256, 255, 253, 256, 256, 256, 253, 256, 256, 255, 256, 256, 256, 256, 254, 256, 253, 256, 64, 16, 4, 1 |
| cbe_target | 512 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D256.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · CBE-DMRG · D=512

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 4, 16, 64, 256, 510, 509, 509, 512, 510, 510, 511, 510, 510, 510, 512, 510, 511, 510, 510, 510, 512, 508, 511, 510, 511, 508, 511, 256, 64, 16, 4, 1 |
| cbe_target | 1024 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D512.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · CBE-TDVP · D=1024

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 1022, 1024, 1024, 1024, 1021, 1022, 1023, 1023, 1023, 1024, 1023, 1023, 1024, 1023, 1022, 1024, 1024, 1021, 1023, 1022, 1024, 1023, 1021, 1023, 1024, 1024, 1022, 256, 16, 1 |
| cbe_target | 1152 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 1024 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D1024.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · CBE-TDVP · D=256

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 255, 255, 254, 256, 255, 255, 256, 256, 256, 255, 255, 256, 255, 255, 254, 256, 256, 256, 256, 254, 255, 253, 253, 254, 254, 255, 256, 256, 16, 1 |
| cbe_target | 288 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D256.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### Hubbard / Z2SU2 · CBE-TDVP · D=512

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 16, 256, 510, 510, 512, 510, 509, 512, 511, 512, 510, 512, 510, 510, 511, 512, 511, 512, 510, 512, 510, 512, 511, 512, 509, 510, 510, 511, 512, 256, 16, 1 |
| cbe_target | 576 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D512.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · 2-DMRG · D=128

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 3, 9, 26, 72, 128, 128, 128, 128, 128, 128, 126, 127, 126, 128, 124, 127, 127, 126, 126, 126, 128, 128, 128, 125, 126, 128, 127, 74, 27, 9, 3, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D128.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · 2-DMRG · D=256

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 3, 9, 26, 72, 192, 253, 256, 255, 255, 255, 256, 255, 254, 255, 255, 255, 255, 256, 253, 256, 254, 253, 254, 256, 254, 254, 192, 72, 26, 9, 3, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D256.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · 2-DMRG · D=512

One complete left-to-right and right-to-left 2-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 3, 9, 27, 80, 232, 509, 510, 512, 511, 511, 511, 511, 511, 512, 512, 510, 512, 512, 509, 512, 512, 509, 510, 510, 509, 511, 218, 80, 27, 9, 3, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D512.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · 2-TDVP · D=128

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 9, 81, 127, 128, 126, 128, 126, 128, 128, 128, 126, 128, 127, 128, 128, 128, 128, 128, 126, 126, 128, 128, 127, 125, 128, 125, 128, 128, 128, 81, 9, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D128.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · 2-TDVP · D=256

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 9, 81, 254, 256, 256, 256, 256, 253, 256, 256, 256, 253, 256, 256, 256, 255, 256, 256, 255, 254, 256, 256, 256, 255, 256, 256, 256, 256, 254, 81, 9, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D256.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · 2-TDVP · D=512

One complete left-to-right and right-to-left 2-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 9, 81, 510, 508, 511, 511, 509, 510, 512, 511, 509, 510, 511, 511, 510, 510, 510, 510, 510, 512, 510, 511, 510, 511, 510, 511, 510, 510, 510, 81, 9, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D512.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · CBE-DMRG · D=1024

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 3, 9, 27, 81, 242, 676, 1023, 1024, 1024, 1024, 1022, 1024, 1019, 1024, 1020, 1024, 1024, 1024, 1024, 1024, 1023, 1023, 1024, 1024, 1023, 672, 242, 81, 27, 9, 3, 1 |
| cbe_target | 2048 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 1024 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D1024.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · CBE-DMRG · D=256

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 3, 9, 26, 72, 192, 253, 256, 255, 255, 255, 256, 255, 254, 255, 255, 255, 255, 256, 253, 256, 254, 253, 254, 256, 254, 254, 192, 72, 26, 9, 3, 1 |
| cbe_target | 512 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D256.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · CBE-DMRG · D=512

One complete left-to-right and right-to-left CBE-DMRG sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 3, 9, 27, 80, 232, 509, 510, 512, 511, 511, 511, 511, 511, 512, 512, 510, 512, 512, 509, 512, 512, 509, 510, 510, 509, 511, 218, 80, 27, 9, 3, 1 |
| cbe_target | 1024 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D512.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · CBE-TDVP · D=1024

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 9, 81, 729, 1024, 1022, 1021, 1023, 1022, 1021, 1024, 1022, 1023, 1021, 1022, 1023, 1023, 1021, 1022, 1022, 1023, 1021, 1022, 1023, 1023, 1021, 1021, 1022, 1024, 729, 81, 9, 1 |
| cbe_target | 1152 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 1024 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D1024.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · CBE-TDVP · D=256

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 9, 81, 254, 256, 256, 256, 256, 253, 256, 256, 256, 253, 256, 256, 256, 255, 256, 256, 255, 254, 256, 256, 256, 255, 256, 256, 256, 256, 254, 81, 9, 1 |
| cbe_target | 288 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 256 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D256.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | One sample at the smallest input for this operation |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

### t-t′-J-J′ / U1SU2 · CBE-TDVP · D=512

One complete left-to-right and right-to-left CBE-TDVP sweep.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 4 |
| bond_dimensions | 1, 9, 81, 510, 508, 511, 511, 509, 510, 512, 511, 509, 510, 511, 511, 510, 510, 510, 510, 510, 512, 510, 511, 510, 511, 510, 511, 510, 510, 510, 81, 9, 1 |
| cbe_target | 576 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 512 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup and measurement |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D512.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 1 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 1 |
| Time budget per case | 600 seconds |
| Compilation warmup | Reused from the smallest input for this operation in the same Julia process |
| Operation mutates its input | Yes |
| Garbage collection before each trial | No |
| Garbage collection after each completed case | Yes |
| Garbage collection before each sample | No |
| Timing overhead correction | 0 nanoseconds |
| blas_threads | 1 |
| gc_threads | 1 |
| julia_threads | 4 |

## Software versions

| Software | Version |
| --- | --- |
| Timing tools (BenchmarkTools) | 1.6.0 |
| Matrix product state library (FiniteMPS) | 2.0.0 |
| Matrix factorization library (MatrixAlgebraKit) | 0.6.9 |
| Tensor computation library (TensorKit) | 0.17.2 |
| Tensor contraction library (TensorOperations) | 5.8.1 |

Complete case identifiers and original parameters are available in the [raw data file](report.json).
