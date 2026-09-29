# Performance measurement report

[View measured source](https://github.com/Qiaoyi-Li/FiniteMPS.jl/commit/9c9f59b69914e9096191d9295449d06d7fee2f44) · [Download raw data (JSON)](report.json) · [View workflow run](https://github.com/Qiaoyi-Li/FiniteMPS.jl/actions/runs/36596136560)

## Measurement environment and thread settings

| Field | Recorded at measurement time |
| --- | --- |
| Repository | Qiaoyi-Li/FiniteMPS.jl |
| Source commit | 9c9f59b69914e9096191d9295449d06d7fee2f44 |
| Benchmark definition commit | 9c9f59b69914e9096191d9295449d06d7fee2f44 |
| Uncommitted changes | No |
| Version tag | Not recorded |
| Release type | Development or local build |
| Algorithm library version | 1.8.3 |
| Measured at (UTC) | 2026-09-29T19:43:49.850Z |
| Processor model | AMD EPYC 9V45 96-Core Processor |
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
| Runner image version | 20260920.314.1 |
| Operating system | Linux |
| System kernel | 6.17.0-1022-azure |
| Visible system memory (bytes) | 16766414848 |
| Run trigger | Commit push |
| Automation workflow | Performance |
| Run identifier | 36596136560 |
| Workflow run number | 1 |
| Run attempt | 1 |
| Benchmark definition file | benchmark/benchmarks.jl |
| Benchmark definition source | Current source checkout |

Processor counts and thread settings describe available resources, not runtime core utilization. System memory is the visible capacity; allocated memory is the amount allocated by the measured operation. Neither value is peak process memory.

## Measurements

| Operation and size | Median time | Total allocated bytes | Memory allocation count | Samples |
| --- | ---: | ---: | ---: | ---: |
| Hubbard / U1SU2 · 2-DMRG · D=128 | 8.6061 seconds | 19678971920 | 183493104 | 3 |
| Hubbard / U1SU2 · 2-DMRG · D=256 | 11.426 seconds | 25288836800 | 224665749 | 3 |
| Hubbard / U1SU2 · 2-DMRG · D=64 | 6.6181 seconds | 13853747784 | 133637772 | 3 |
| Hubbard / U1SU2 · 2-TDVP · D=128 | 143.92 seconds | 206152120496 | 1717526267 | 3 |
| Hubbard / U1SU2 · 2-TDVP · D=256 | 242.15 seconds | 449493733768 | 3466753869 | 3 |
| Hubbard / U1SU2 · 2-TDVP · D=64 | 57.142 seconds | 83427891344 | 814043077 | 3 |
| Hubbard / U1SU2 · CBE-DMRG · D=128 | 3.6075 seconds | 7455054344 | 74487341 | 3 |
| Hubbard / U1SU2 · CBE-DMRG · D=256 | 4.9027 seconds | 9887245344 | 92993741 | 3 |
| Hubbard / U1SU2 · CBE-DMRG · D=64 | 3.0599 seconds | 5540731960 | 56859957 | 3 |
| Hubbard / U1SU2 · CBE-TDVP · D=128 | 13.175 seconds | 31273556720 | 264775138 | 3 |
| Hubbard / U1SU2 · CBE-TDVP · D=256 | 18.254 seconds | 39120891496 | 318274214 | 3 |
| Hubbard / U1SU2 · CBE-TDVP · D=64 | 8.775 seconds | 19577883968 | 171764891 | 3 |
| Hubbard / U1U1 · 2-DMRG · D=128 | 7.8834 seconds | 15541062328 | 167196870 | 3 |
| Hubbard / U1U1 · 2-DMRG · D=256 | 12.455 seconds | 26357253536 | 198151044 | 3 |
| Hubbard / U1U1 · 2-DMRG · D=64 | 6.0656 seconds | 11824156544 | 148857779 | 3 |
| Hubbard / U1U1 · 2-TDVP · D=128 | 117.57 seconds | 239802762096 | 1974472143 | 3 |
| Hubbard / U1U1 · 2-TDVP · D=256 | 218.45 seconds | 445477689528 | 2758466970 | 3 |
| Hubbard / U1U1 · 2-TDVP · D=64 | 76.353 seconds | 162672208768 | 1568832427 | 3 |
| Hubbard / U1U1 · CBE-DMRG · D=128 | 2.5523 seconds | 4698107976 | 45254431 | 3 |
| Hubbard / U1U1 · CBE-DMRG · D=256 | 4.3878 seconds | 9332612536 | 52462559 | 3 |
| Hubbard / U1U1 · CBE-DMRG · D=64 | 2.0831 seconds | 3050890544 | 38247898 | 3 |
| Hubbard / U1U1 · CBE-TDVP · D=128 | 6.823 seconds | 13558433960 | 127168673 | 3 |
| Hubbard / U1U1 · CBE-TDVP · D=256 | 12.822 seconds | 27558690560 | 163926596 | 3 |
| Hubbard / U1U1 · CBE-TDVP · D=64 | 5.2212 seconds | 8872600728 | 106127439 | 3 |
| Hubbard / Z2SU2 · 2-DMRG · D=128 | 2.3602 seconds | 3337279368 | 33153893 | 3 |
| Hubbard / Z2SU2 · 2-DMRG · D=256 | 4.4781 seconds | 6644660144 | 41237131 | 3 |
| Hubbard / Z2SU2 · 2-DMRG · D=64 | 1.4786 seconds | 2139526400 | 27000439 | 3 |
| Hubbard / Z2SU2 · 2-TDVP · D=128 | 22.307 seconds | 39286334376 | 187685079 | 3 |
| Hubbard / Z2SU2 · 2-TDVP · D=256 | 71.742 seconds | 106739939792 | 271519707 | 3 |
| Hubbard / Z2SU2 · 2-TDVP · D=64 | 9.252 seconds | 17057564416 | 122192278 | 3 |
| Hubbard / Z2SU2 · CBE-DMRG · D=128 | 1.1305 seconds | 1446917024 | 12482581 | 3 |
| Hubbard / Z2SU2 · CBE-DMRG · D=256 | 1.6992 seconds | 2925915072 | 14058338 | 3 |
| Hubbard / Z2SU2 · CBE-DMRG · D=64 | 831.6 milliseconds | 861620248 | 10855686 | 3 |
| Hubbard / Z2SU2 · CBE-TDVP · D=128 | 2.3056 seconds | 4063099048 | 24964615 | 3 |
| Hubbard / Z2SU2 · CBE-TDVP · D=256 | 5.6753 seconds | 10961781152 | 36423581 | 3 |
| Hubbard / Z2SU2 · CBE-TDVP · D=64 | 1.3632 seconds | 1770681824 | 16940335 | 3 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=128 | 5.0403 seconds | 11255611368 | 112522715 | 3 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=256 | 6.9013 seconds | 13687457080 | 128210307 | 3 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=64 | 3.9315 seconds | 8915264888 | 93472905 | 3 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=128 | 75.74 seconds | 155103743080 | 1343423244 | 3 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=256 | 79.201 seconds | 161296269232 | 1349207663 | 3 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=64 | 44.69 seconds | 97338884560 | 840801813 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=128 | 2.8545 seconds | 5279412896 | 54680564 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=256 | 3.5135 seconds | 7394521248 | 68887904 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=64 | 2.162 seconds | 4070034616 | 44285949 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=128 | 11.872 seconds | 25763992304 | 229433772 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=256 | 14.003 seconds | 29422239344 | 249294095 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=64 | 6.4835 seconds | 12612438584 | 113494650 | 3 |

### Hubbard / U1SU2 · 2-DMRG · D=128

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D128.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1SU2 · 2-DMRG · D=256

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D256.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1SU2 · 2-DMRG · D=64

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 4, 16, 64, 64, 63, 64, 64, 64, 62, 64, 64, 63, 64, 62, 63, 64, 64, 62, 63, 63, 62, 63, 64, 64, 63, 62, 62, 63, 57, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D64.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1SU2 · 2-TDVP · D=128

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D128.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1SU2 · 2-TDVP · D=256

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D256.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1SU2 · 2-TDVP · D=64

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 16, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 63, 63, 64, 63, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D64.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1SU2 · CBE-DMRG · D=128

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 4, 16, 64, 126, 126, 128, 128, 128, 127, 125, 128, 126, 128, 127, 128, 128, 128, 128, 128, 128, 126, 125, 127, 128, 128, 127, 127, 126, 64, 16, 4, 1 |
| cbe_target | 256 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D128.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D256.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1SU2 · CBE-DMRG · D=64

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 4, 16, 64, 64, 63, 64, 64, 64, 62, 64, 64, 63, 64, 62, 63, 64, 64, 62, 63, 63, 62, 63, 64, 64, 63, 62, 62, 63, 57, 16, 4, 1 |
| cbe_target | 128 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_ground_D64.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1SU2 · CBE-TDVP · D=128

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 16, 127, 128, 127, 128, 125, 128, 126, 128, 127, 126, 126, 126, 127, 128, 126, 128, 127, 128, 126, 128, 127, 127, 126, 126, 126, 128, 128, 128, 127, 16, 1 |
| cbe_target | 144 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D128.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D256.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1SU2 · CBE-TDVP · D=64

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 16, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 63, 63, 64, 63, 16, 1 |
| cbe_target | 72 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_u1su2 |
| model_name | Hubbard / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1su2_thermal_D64.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1U1 · 2-DMRG · D=128

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D128.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1U1 · 2-DMRG · D=256

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D256.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1U1 · 2-DMRG · D=64

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 4, 16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 58, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D64.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1U1 · 2-TDVP · D=128

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D128.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1U1 · 2-TDVP · D=256

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D256.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1U1 · 2-TDVP · D=64

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D64.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1U1 · CBE-DMRG · D=128

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 4, 16, 64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 16, 4, 1 |
| cbe_target | 256 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D128.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D256.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1U1 · CBE-DMRG · D=64

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 4, 16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 58, 16, 4, 1 |
| cbe_target | 128 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_ground_D64.json |
| state | ground |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1U1 · CBE-TDVP · D=128

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 16, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 16, 1 |
| cbe_target | 144 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D128.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D256.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / U1U1 · CBE-TDVP · D=64

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 16, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 16, 1 |
| cbe_target | 72 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_u1u1 |
| model_name | Hubbard / U1U1 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_u1u1_thermal_D64.json |
| state | thermal |
| symmetry | U1U1 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / Z2SU2 · 2-DMRG · D=128

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D128.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / Z2SU2 · 2-DMRG · D=256

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D256.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / Z2SU2 · 2-DMRG · D=64

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 4, 16, 64, 64, 63, 62, 62, 64, 63, 64, 64, 64, 64, 62, 64, 62, 64, 64, 61, 63, 63, 60, 63, 63, 64, 63, 64, 61, 64, 16, 4, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D64.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / Z2SU2 · 2-TDVP · D=128

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D128.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / Z2SU2 · 2-TDVP · D=256

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D256.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / Z2SU2 · 2-TDVP · D=64

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 16, 63, 63, 64, 64, 64, 64, 64, 64, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 63, 63, 16, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D64.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / Z2SU2 · CBE-DMRG · D=128

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 4, 16, 64, 128, 128, 128, 128, 127, 128, 128, 128, 127, 128, 128, 128, 127, 128, 128, 128, 127, 128, 128, 128, 127, 128, 128, 128, 128, 64, 16, 4, 1 |
| cbe_target | 256 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D128.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D256.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / Z2SU2 · CBE-DMRG · D=64

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 4, 16, 64, 64, 63, 62, 62, 64, 63, 64, 64, 64, 64, 62, 64, 62, 64, 64, 61, 63, 63, 60, 63, 63, 64, 63, 64, 61, 64, 16, 4, 1 |
| cbe_target | 128 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_ground_D64.json |
| state | ground |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / Z2SU2 · CBE-TDVP · D=128

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 16, 127, 128, 127, 128, 128, 126, 126, 126, 127, 126, 128, 126, 127, 126, 126, 126, 127, 126, 126, 126, 125, 128, 126, 126, 127, 128, 128, 128, 127, 16, 1 |
| cbe_target | 144 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D128.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D256.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### Hubbard / Z2SU2 · CBE-TDVP · D=64

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 16, 63, 63, 64, 64, 64, 64, 64, 64, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 63, 63, 16, 1 |
| cbe_target | 72 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | hubbard_z2su2 |
| model_name | Hubbard / Z2SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/hubbard_z2su2_thermal_D64.json |
| state | thermal |
| symmetry | Z2SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### t-t′-J-J′ / U1SU2 · 2-DMRG · D=128

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D128.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### t-t′-J-J′ / U1SU2 · 2-DMRG · D=256

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D256.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### t-t′-J-J′ / U1SU2 · 2-DMRG · D=64

One complete left-to-right and right-to-left 2-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 3, 9, 24, 36, 63, 42, 63, 42, 63, 42, 64, 42, 61, 63, 61, 59, 63, 62, 64, 63, 64, 64, 61, 63, 64, 63, 61, 57, 27, 9, 3, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | Not recorded |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | 2-DMRG |
| rsvd | No |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D64.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### t-t′-J-J′ / U1SU2 · 2-TDVP · D=128

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D128.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### t-t′-J-J′ / U1SU2 · 2-TDVP · D=256

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D256.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### t-t′-J-J′ / U1SU2 · 2-TDVP · D=64

One complete left-to-right and right-to-left 2-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 9, 63, 64, 64, 64, 64, 63, 64, 63, 64, 64, 64, 64, 64, 63, 64, 63, 64, 64, 64, 64, 64, 63, 64, 63, 64, 64, 64, 63, 63, 9, 1 |
| cbe_target | Not recorded |
| cbe_tolerance | Not recorded |
| continuation_after | Not recorded |
| dt | -0.1 |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | 2-TDVP |
| rsvd | No |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D64.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### t-t′-J-J′ / U1SU2 · CBE-DMRG · D=128

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 3, 9, 26, 72, 128, 128, 128, 128, 128, 128, 126, 127, 126, 128, 124, 127, 127, 126, 126, 126, 128, 128, 128, 125, 126, 128, 127, 74, 27, 9, 3, 1 |
| cbe_target | 256 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D128.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D256.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### t-t′-J-J′ / U1SU2 · CBE-DMRG · D=64

One complete left-to-right and right-to-left CBE-DMRG sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 3, 9, 24, 36, 63, 42, 63, 42, 63, 42, 64, 42, 61, 63, 61, 59, 63, 62, 64, 63, 64, 64, 61, 63, 64, 63, 61, 57, 27, 9, 3, 1 |
| cbe_target | 128 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-DMRG |
| dt | Not recorded |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | CBE-DMRG |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_ground_D64.json |
| state | ground |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### t-t′-J-J′ / U1SU2 · CBE-TDVP · D=128

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 9, 81, 127, 128, 126, 128, 126, 128, 128, 128, 126, 128, 127, 128, 128, 128, 128, 128, 126, 126, 128, 128, 127, 125, 128, 125, 128, 128, 128, 81, 9, 1 |
| cbe_target | 144 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 128 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D128.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
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
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D256.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 0 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

### t-t′-J-J′ / U1SU2 · CBE-TDVP · D=64

One complete left-to-right and right-to-left CBE-TDVP sweep. Samples continue the same state; the CBE algorithm follows its two-site partner.


#### Workload details

| Field | Value |
| --- | --- |
| GCstep | No |
| GCsweep | No |
| K | 8 |
| bond_dimensions | 1, 9, 63, 64, 64, 64, 64, 63, 64, 63, 64, 64, 64, 64, 64, 63, 64, 63, 64, 64, 64, 64, 64, 63, 64, 63, 64, 64, 64, 63, 63, 9, 1 |
| cbe_target | 72 |
| cbe_tolerance | 1.0e-8 |
| continuation_after | 2-TDVP |
| dt | -0.1 |
| model | tj_u1su2 |
| model_name | t-t′-J-J′ / U1SU2 |
| model_parameters | Structured parameter; see the raw data |
| nominal_D | 64 |
| operation | CBE-TDVP |
| rsvd | Yes |
| sampling_state | continue across warmup, samples and paired algorithms |
| scalar_type | Float64 |
| sector_preset | presets/tj_u1su2_thermal_D64.json |
| state | thermal |
| symmetry | U1SU2 |
| truncation | truncrank(D) |

#### Sampling and execution settings

| Field | Value |
| --- | --- |
| Collected samples | 3 |
| Evaluations per sample | 1 |
| Warmup samples | 1 |
| Random seed | 20260929 |
| Sample limit | 3 |
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

## Software versions

| Software | Version |
| --- | --- |
| Timing tools (BenchmarkTools) | 1.6.0 |
| Matrix product state library (FiniteMPS) | 1.8.3 |
| Matrix factorization library (MatrixAlgebraKit) | 0.6.9 |
| Tensor computation library (TensorKit) | 0.17.2 |
| Tensor contraction library (TensorOperations) | 5.8.1 |

Complete case identifiers and original parameters are available in the [raw data file](report.json).
