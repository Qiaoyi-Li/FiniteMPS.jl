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
| Measured at (UTC) | 2026-09-29T18:12:57.224Z |
| Processor model | AMD EPYC 9V45 96-Core Processor |
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
| Run identifier | 36596136560 |
| Workflow run number | 1 |
| Run attempt | 1 |
| Benchmark definition file | benchmark/benchmarks.jl |
| Benchmark definition source | Current source checkout |

Processor counts and thread settings describe available resources, not runtime core utilization. System memory is the visible capacity; allocated memory is the amount allocated by the measured operation. Neither value is peak process memory.

## Measurements

| Operation and size | Median time | Total allocated bytes | Memory allocation count | Samples |
| --- | ---: | ---: | ---: | ---: |
| Hubbard / U1SU2 · 2-DMRG · D=128 | 9.1745 seconds | 16435406080 | 156478894 | 3 |
| Hubbard / U1SU2 · 2-DMRG · D=256 | 11.989 seconds | 21290597928 | 190856622 | 3 |
| Hubbard / U1SU2 · 2-DMRG · D=64 | 6.5942 seconds | 11664297536 | 115665584 | 3 |
| Hubbard / U1SU2 · 2-TDVP · D=128 | 147.53 seconds | 179156799672 | 1524674927 | 3 |
| Hubbard / U1SU2 · 2-TDVP · D=256 | 244.61 seconds | 365174173144 | 2858951973 | 3 |
| Hubbard / U1SU2 · 2-TDVP · D=64 | 61.482 seconds | 80320397344 | 791825378 | 3 |
| Hubbard / U1SU2 · CBE-DMRG · D=128 | 3.9063 seconds | 6127496864 | 61818655 | 3 |
| Hubbard / U1SU2 · CBE-DMRG · D=256 | 4.8761 seconds | 8175980928 | 76419429 | 3 |
| Hubbard / U1SU2 · CBE-DMRG · D=64 | 3.0332 seconds | 4562472496 | 47725763 | 3 |
| Hubbard / U1SU2 · CBE-TDVP · D=128 | 13.58 seconds | 26288664344 | 223914510 | 3 |
| Hubbard / U1SU2 · CBE-TDVP · D=256 | 17.763 seconds | 33072074792 | 268226719 | 3 |
| Hubbard / U1SU2 · CBE-TDVP · D=64 | 8.6973 seconds | 16604103928 | 147551547 | 3 |
| Hubbard / U1U1 · 2-DMRG · D=128 | 9.0965 seconds | 15472856896 | 167106380 | 3 |
| Hubbard / U1U1 · 2-DMRG · D=256 | 15.182 seconds | 26280672704 | 198105976 | 3 |
| Hubbard / U1U1 · 2-DMRG · D=64 | 7.1434 seconds | 11771100024 | 148792676 | 3 |
| Hubbard / U1U1 · 2-TDVP · D=128 | 135.17 seconds | 238772442064 | 1974332439 | 3 |
| Hubbard / U1U1 · 2-TDVP · D=256 | 258.33 seconds | 444512248632 | 2758312015 | 3 |
| Hubbard / U1U1 · 2-TDVP · D=64 | 84.354 seconds | 162488082096 | 1568753816 | 3 |
| Hubbard / U1U1 · CBE-DMRG · D=128 | 2.9875 seconds | 4656732048 | 45157032 | 3 |
| Hubbard / U1U1 · CBE-DMRG · D=256 | 5.367 seconds | 9289874192 | 52385265 | 3 |
| Hubbard / U1U1 · CBE-DMRG · D=64 | 2.3938 seconds | 3015910320 | 38155925 | 3 |
| Hubbard / U1U1 · CBE-TDVP · D=128 | 7.3552 seconds | 13481200104 | 127071278 | 3 |
| Hubbard / U1U1 · CBE-TDVP · D=256 | 15.872 seconds | 27465624552 | 163706020 | 3 |
| Hubbard / U1U1 · CBE-TDVP · D=64 | 5.3857 seconds | 8842749760 | 105985595 | 3 |
| Hubbard / Z2SU2 · 2-DMRG · D=128 | 2.6876 seconds | 3297307688 | 32816455 | 3 |
| Hubbard / Z2SU2 · 2-DMRG · D=256 | 4.6904 seconds | 6344598688 | 38786200 | 3 |
| Hubbard / Z2SU2 · 2-DMRG · D=64 | 1.5743 seconds | 2138798064 | 26991825 | 3 |
| Hubbard / Z2SU2 · 2-TDVP · D=128 | 26.143 seconds | 38513431968 | 182556233 | 3 |
| Hubbard / Z2SU2 · 2-TDVP · D=256 | 84.089 seconds | 103446824936 | 249739395 | 3 |
| Hubbard / Z2SU2 · 2-TDVP · D=64 | 10.79 seconds | 17059935440 | 122202437 | 3 |
| Hubbard / Z2SU2 · CBE-DMRG · D=128 | 1.058 seconds | 1400389080 | 11965137 | 3 |
| Hubbard / Z2SU2 · CBE-DMRG · D=256 | 1.828 seconds | 2841610280 | 13149646 | 3 |
| Hubbard / Z2SU2 · CBE-DMRG · D=64 | 751.83 milliseconds | 820547616 | 10396618 | 3 |
| Hubbard / Z2SU2 · CBE-TDVP · D=128 | 2.5189 seconds | 3912916152 | 23661266 | 3 |
| Hubbard / Z2SU2 · CBE-TDVP · D=256 | 7.048 seconds | 10528325432 | 33075898 | 3 |
| Hubbard / Z2SU2 · CBE-TDVP · D=64 | 1.4023 seconds | 1764525104 | 16878465 | 3 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=128 | 5.5633 seconds | 9731893248 | 100055571 | 3 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=256 | 7.6553 seconds | 11993257088 | 114195067 | 3 |
| t-t′-J-J′ / U1SU2 · 2-DMRG · D=64 | 4.2983 seconds | 7727338224 | 83906410 | 3 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=128 | 73.615 seconds | 131621569296 | 1165290025 | 3 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=256 | 80.579 seconds | 139584710024 | 1184623265 | 3 |
| t-t′-J-J′ / U1SU2 · 2-TDVP · D=64 | 50.023 seconds | 82114787264 | 727369506 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=128 | 3.0854 seconds | 4723901328 | 49450167 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=256 | 3.9843 seconds | 6638917488 | 61780105 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-DMRG · D=64 | 2.4055 seconds | 3635695112 | 40255080 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=128 | 11.833 seconds | 22608108224 | 203256590 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=256 | 12.834 seconds | 25893109904 | 220335931 | 3 |
| t-t′-J-J′ / U1SU2 · CBE-TDVP · D=64 | 6.9181 seconds | 11147911120 | 101572184 | 3 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
| julia_threads | 2 |

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
