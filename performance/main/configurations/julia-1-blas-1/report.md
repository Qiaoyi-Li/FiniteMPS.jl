# Performance measurement report

[View measured source](https://github.com/Qiaoyi-Li/FiniteMPS.jl/commit/a5d972830d50d43a5ddfa58242ee44556cad2b9e) · [Download raw data (JSON)](report.json) · [View workflow run](https://github.com/Qiaoyi-Li/FiniteMPS.jl/actions/runs/37001879445)

## Measurement environment and thread settings

| Field | Recorded at measurement time |
| --- | --- |
| Repository | Qiaoyi-Li/FiniteMPS.jl |
| Source commit | a5d972830d50d43a5ddfa58242ee44556cad2b9e |
| Benchmark definition commit | a5d972830d50d43a5ddfa58242ee44556cad2b9e |
| Uncommitted changes | No |
| Version tag | Not recorded |
| Release type | Development or local build |
| Algorithm library version | 1.8.3 |
| Measured at (UTC) | 2026-10-02T11:49:08.738Z |
| Processor model | AMD EPYC 7763 64-Core Processor |
| Processor architecture | 64-bit x86 |
| Visible logical processors | 4 |
| Processors available to this process | 4 |
| Allowed processor identifiers | 0-3 |
| Visible physical cores | 2 |
| Julia version | 1.11.6 |
| Julia computation threads | 1 |
| Julia interactive threads | 0 |
| Garbage collection threads | 1 |
| Matrix computation threads | 1 |
| Matrix computation backend | OpenBLAS |
| Runner label | ubuntu-24.04 |
| Runner type | GitHub-hosted runner |
| Runner image version | 20260927.320.1 |
| Operating system | Linux |
| System kernel | 6.17.0-1022-azure |
| Visible system memory (bytes) | 16766414848 |
| Run trigger | Commit push |
| Automation workflow | Performance |
| Run identifier | 37001879445 |
| Workflow run number | 3 |
| Run attempt | 1 |
| Benchmark definition file | benchmark/benchmarks.jl |
| Benchmark definition source | Current source checkout |

Processor counts and thread settings describe available resources, not runtime core utilization. System memory is the visible capacity; allocated memory is the amount allocated by the measured operation. Neither value is peak process memory.

## Measurements

| Operation and size | Median time | Total allocated bytes | Memory allocation count | Samples |
| --- | ---: | ---: | ---: | ---: |
| Hubbard / U1SU2 · 2-DMRG · D=128 | 13.109 seconds | 12234767752 | 110253109 | 1 |
| Hubbard / U1SU2 · 2-DMRG · D=256 | 15.637 seconds | 14489846520 | 123477538 | 1 |
| Hubbard / U1SU2 · 2-DMRG · D=512 | 22.46 seconds | 19157930400 | 144622548 | 1 |
| Hubbard / U1SU2 · 2-TDVP · D=128 | 137.98 seconds | 116566543584 | 953725882 | 1 |
| Hubbard / U1SU2 · 2-TDVP · D=256 | 352.82 seconds | 317057809744 | 2396500592 | 1 |
| Hubbard / U1SU2 · 2-TDVP · D=512 | 492.31 seconds | 398722497800 | 2791203316 | 1 |
| Hubbard / U1SU2 · CBE-DMRG · D=1024 | 29.298 seconds | 18118455896 | 73917820 | 1 |
| Hubbard / U1SU2 · CBE-DMRG · D=256 | 6.8582 seconds | 6891055504 | 63598566 | 1 |
| Hubbard / U1SU2 · CBE-DMRG · D=512 | 10.654 seconds | 10005325040 | 76994532 | 1 |
| Hubbard / U1SU2 · CBE-TDVP · D=1024 | 92.723 seconds | 66456786728 | 372314263 | 1 |
| Hubbard / U1SU2 · CBE-TDVP · D=256 | 28.898 seconds | 30269723944 | 243818014 | 1 |
| Hubbard / U1SU2 · CBE-TDVP · D=512 | 40.764 seconds | 38340543640 | 273956177 | 1 |
| Hubbard / U1U1 · 2-DMRG · D=128 | 11.249 seconds | 8423439176 | 90152091 | 1 |
| Hubbard / U1U1 · 2-DMRG · D=256 | 19.833 seconds | 13120705200 | 88438444 | 1 |
| Hubbard / U1U1 · 2-DMRG · D=512 | 58.259 seconds | 30944554064 | 98939609 | 1 |
| Hubbard / U1U1 · 2-TDVP · D=128 | 143.54 seconds | 121847330096 | 994859653 | 1 |
| Hubbard / U1U1 · 2-TDVP · D=256 | 296.75 seconds | 228517819568 | 1406379622 | 1 |
| Hubbard / U1U1 · 2-TDVP · D=512 | 787.74 seconds | 484489277176 | 1584779840 | 1 |
| Hubbard / U1U1 · CBE-DMRG · D=1024 | 152.5 seconds | 65350463864 | 33856162 | 1 |
| Hubbard / U1U1 · CBE-DMRG · D=256 | 9.4283 seconds | 6292225424 | 34210847 | 1 |
| Hubbard / U1U1 · CBE-DMRG · D=512 | 28.163 seconds | 17159526528 | 37113563 | 1 |
| Hubbard / U1U1 · CBE-TDVP · D=1024 | 309.31 seconds | 137932667240 | 141145129 | 1 |
| Hubbard / U1U1 · CBE-TDVP · D=256 | 24.669 seconds | 17831574216 | 111839675 | 1 |
| Hubbard / U1U1 · CBE-TDVP · D=512 | 73.528 seconds | 43373141728 | 122593337 | 1 |
| Hubbard / Z2SU2 · 2-DMRG · D=128 | 3.6339 seconds | 1822748272 | 16636629 | 1 |
| Hubbard / Z2SU2 · 2-DMRG · D=256 | 7.2176 seconds | 3533914120 | 18691282 | 1 |
| Hubbard / Z2SU2 · 2-DMRG · D=512 | 23.29 seconds | 9327028584 | 22489870 | 1 |
| Hubbard / Z2SU2 · 2-TDVP · D=128 | 35.902 seconds | 21001836088 | 100443563 | 1 |
| Hubbard / Z2SU2 · 2-TDVP · D=256 | 123.66 seconds | 62185392896 | 185836453 | 1 |
| Hubbard / Z2SU2 · 2-TDVP · D=512 | 449.25 seconds | 161306167648 | 205958415 | 1 |
| Hubbard / Z2SU2 · CBE-DMRG · D=1024 | 81.76 seconds | 25684693360 | 9337151 | 1 |
| Hubbard / Z2SU2 · CBE-DMRG · D=256 | 4.0399 seconds | 2369078784 | 8931203 | 1 |
| Hubbard / Z2SU2 · CBE-DMRG · D=512 | 14.19 seconds | 6839879184 | 10387621 | 1 |
| Hubbard / Z2SU2 · CBE-TDVP · D=1024 | 237.86 seconds | 71421418136 | 39958028 | 1 |
| Hubbard / Z2SU2 · CBE-TDVP · D=256 | 12.668 seconds | 7135532928 | 21591170 | 1 |
| Hubbard / Z2SU2 · CBE-TDVP · D=512 | 49.051 seconds | 20735036824 | 28720060 | 1 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=128 | 8.0179 seconds | 7235516912 | 70045588 | 1 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=256 | 10.757 seconds | 8345098984 | 75594068 | 1 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=512 | 14.724 seconds | 11513470200 | 88623371 | 1 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=128 | 98.307 seconds | 86872137544 | 743159238 | 1 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=256 | 123.34 seconds | 105455383728 | 863255905 | 1 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=512 | 202.08 seconds | 158809042576 | 1152669122 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=1024 | 24.249 seconds | 15259359000 | 55220131 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=256 | 5.9684 seconds | 5449820960 | 49635236 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=512 | 9.7292 seconds | 7993745120 | 57174790 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=1024 | 72.963 seconds | 53890772712 | 315041390 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=256 | 20.287 seconds | 19940195416 | 165821708 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=512 | 30.855 seconds | 27715413096 | 192048165 | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

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
| julia_threads | 1 |

## Software versions

| Software | Version |
| --- | --- |
| Timing tools (BenchmarkTools) | 1.6.0 |
| Matrix product state library (FiniteMPS) | 1.8.3 |
| Matrix factorization library (MatrixAlgebraKit) | 0.6.9 |
| Tensor computation library (TensorKit) | 0.17.2 |
| Tensor contraction library (TensorOperations) | 5.8.1 |

Complete case identifiers and original parameters are available in the [raw data file](report.json).
