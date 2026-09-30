# Performance measurement report

[View measured source](https://github.com/Qiaoyi-Li/FiniteMPS.jl/commit/5ea641a53f5176f1bd676ad08e1f037c0706c9b5) · [Download raw data (JSON)](report.json) · [View workflow run](https://github.com/Qiaoyi-Li/FiniteMPS.jl/actions/runs/36639230982)

## Measurement environment and thread settings

| Field | Recorded at measurement time |
| --- | --- |
| Repository | Qiaoyi-Li/FiniteMPS.jl |
| Source commit | 5ea641a53f5176f1bd676ad08e1f037c0706c9b5 |
| Benchmark definition commit | 5ea641a53f5176f1bd676ad08e1f037c0706c9b5 |
| Uncommitted changes | No |
| Version tag | Not recorded |
| Release type | Development or local build |
| Algorithm library version | 1.8.3 |
| Measured at (UTC) | 2026-09-30T00:30:33.482Z |
| Processor model | AMD EPYC 9V74 80-Core Processor |
| Processor architecture | 64-bit x86 |
| Visible logical processors | 4 |
| Processors available to this process | 4 |
| Allowed processor identifiers | 0-3 |
| Visible physical cores | 2 |
| Julia version | 1.11.6 |
| Julia computation threads | 2 |
| Julia interactive threads | 0 |
| Garbage collection threads | 1 |
| Matrix computation threads | 1 |
| Matrix computation backend | OpenBLAS |
| Runner label | ubuntu-24.04 |
| Runner type | GitHub-hosted runner |
| Runner image version | 20260920.314.1 |
| Operating system | Linux |
| System kernel | 6.17.0-1022-azure |
| Visible system memory (bytes) | 16766414848 |
| Run trigger | Commit push |
| Automation workflow | Performance |
| Run identifier | 36639230982 |
| Workflow run number | 2 |
| Run attempt | 1 |
| Benchmark definition file | benchmark/benchmarks.jl |
| Benchmark definition source | Current source checkout |

Processor counts and thread settings describe available resources, not runtime core utilization. System memory is the visible capacity; allocated memory is the amount allocated by the measured operation. Neither value is peak process memory.

## Measurements

| Operation and size | Median time | Total allocated bytes | Memory allocation count | Samples |
| --- | ---: | ---: | ---: | ---: |
| Hubbard / U1SU2 · 2-DMRG · D=128 | 9.8906 seconds | 13952634864 | 125226184 | 1 |
| Hubbard / U1SU2 · 2-DMRG · D=256 | 11.593 seconds | 16356369232 | 139996050 | 1 |
| Hubbard / U1SU2 · 2-DMRG · D=512 | 16.176 seconds | 21187748296 | 162758459 | 1 |
| Hubbard / U1SU2 · 2-TDVP · D=128 | 102.89 seconds | 133905081832 | 1079212708 | 1 |
| Hubbard / U1SU2 · 2-TDVP · D=256 | 286.02 seconds | 383501257200 | 2880845987 | 1 |
| Hubbard / U1SU2 · 2-TDVP · D=512 | 385.98 seconds | 473102859960 | 3335023777 | 1 |
| Hubbard / U1SU2 · CBE-DMRG · D=1024 | 19.996 seconds | 19148126520 | 84086114 | 1 |
| Hubbard / U1SU2 · CBE-DMRG · D=256 | 5.6039 seconds | 7976959600 | 74378282 | 1 |
| Hubbard / U1SU2 · CBE-DMRG · D=512 | 8.2517 seconds | 11243355600 | 89480398 | 1 |
| Hubbard / U1SU2 · CBE-TDVP · D=1024 | 65.272 seconds | 73026205440 | 430925943 | 1 |
| Hubbard / U1SU2 · CBE-TDVP · D=256 | 23.967 seconds | 35325993240 | 287724241 | 1 |
| Hubbard / U1SU2 · CBE-TDVP · D=512 | 30.892 seconds | 43375849360 | 318058626 | 1 |
| Hubbard / U1U1 · 2-DMRG · D=128 | 7.7272 seconds | 8488936776 | 90707182 | 1 |
| Hubbard / U1U1 · 2-DMRG · D=256 | 12.466 seconds | 13187419024 | 89000203 | 1 |
| Hubbard / U1U1 · 2-DMRG · D=512 | 36.087 seconds | 31020387608 | 99564568 | 1 |
| Hubbard / U1U1 · 2-TDVP · D=128 | 96.902 seconds | 122213347512 | 995690666 | 1 |
| Hubbard / U1U1 · 2-TDVP · D=256 | 196.49 seconds | 229035323840 | 1407351087 | 1 |
| Hubbard / U1U1 · 2-TDVP · D=512 | 487.7 seconds | 484970437968 | 1585692747 | 1 |
| Hubbard / U1U1 · CBE-DMRG · D=1024 | 97.511 seconds | 65387305264 | 34212407 | 1 |
| Hubbard / U1U1 · CBE-DMRG · D=256 | 6.2043 seconds | 6341982024 | 34652275 | 1 |
| Hubbard / U1U1 · CBE-DMRG · D=512 | 17.889 seconds | 17201226256 | 37436448 | 1 |
| Hubbard / U1U1 · CBE-TDVP · D=1024 | 189.51 seconds | 137985922680 | 141620047 | 1 |
| Hubbard / U1U1 · CBE-TDVP · D=256 | 15.786 seconds | 17905588072 | 112292914 | 1 |
| Hubbard / U1U1 · CBE-TDVP · D=512 | 45.673 seconds | 43471613728 | 123082002 | 1 |
| Hubbard / Z2SU2 · 2-DMRG · D=128 | 2.3603 seconds | 1943160296 | 17791322 | 1 |
| Hubbard / Z2SU2 · 2-DMRG · D=256 | 4.3663 seconds | 3732229328 | 20613377 | 1 |
| Hubbard / Z2SU2 · 2-DMRG · D=512 | 13.631 seconds | 9756139968 | 26017097 | 1 |
| Hubbard / Z2SU2 · 2-TDVP · D=128 | 23.121 seconds | 22301908680 | 109416103 | 1 |
| Hubbard / Z2SU2 · 2-TDVP · D=256 | 82.628 seconds | 68793245944 | 229844007 | 1 |
| Hubbard / Z2SU2 · 2-TDVP · D=512 | 284.09 seconds | 166868669000 | 243178190 | 1 |
| Hubbard / Z2SU2 · CBE-DMRG · D=1024 | 49.937 seconds | 25784654960 | 10508525 | 1 |
| Hubbard / Z2SU2 · CBE-DMRG · D=256 | 2.7434 seconds | 2469080528 | 10200134 | 1 |
| Hubbard / Z2SU2 · CBE-DMRG · D=512 | 8.865 seconds | 6949777456 | 11706421 | 1 |
| Hubbard / Z2SU2 · CBE-TDVP · D=1024 | 150.07 seconds | 72504455264 | 48027228 | 1 |
| Hubbard / Z2SU2 · CBE-TDVP · D=256 | 8.3292 seconds | 7676437376 | 25775607 | 1 |
| Hubbard / Z2SU2 · CBE-TDVP · D=512 | 29.551 seconds | 21558773600 | 34933666 | 1 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=128 | 6.6434 seconds | 8062378088 | 77317923 | 1 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=256 | 7.9702 seconds | 9274733208 | 83705558 | 1 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=512 | 10.65 seconds | 12509399392 | 97394204 | 1 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=128 | 76.511 seconds | 100934010552 | 850769408 | 1 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=256 | 93.122 seconds | 121443717352 | 985859280 | 1 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=512 | 144.55 seconds | 180932851224 | 1322729269 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=1024 | 15.707 seconds | 15874344496 | 61216076 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=256 | 5.2093 seconds | 6096391328 | 56049649 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=512 | 7.0221 seconds | 8700523208 | 64297671 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=1024 | 49.849 seconds | 58309052688 | 354018411 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=256 | 15.71 seconds | 22424842560 | 186875834 | 1 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=512 | 22.878 seconds | 30307699000 | 214210023 | 1 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

## Software versions

| Software | Version |
| --- | --- |
| Timing tools (BenchmarkTools) | 1.6.0 |
| Matrix product state library (FiniteMPS) | 1.8.3 |
| Matrix factorization library (MatrixAlgebraKit) | 0.6.9 |
| Tensor computation library (TensorKit) | 0.17.2 |
| Tensor contraction library (TensorOperations) | 5.8.1 |

Complete case identifiers and original parameters are available in the [raw data file](report.json).
