# Performance report

Source commit: `5ea641a53f5176f1bd676ad08e1f037c0706c9b5`

Julia uses 1, 2, or 4 compute threads; the linear algebra backend and garbage collector each use 1 thread.

| Thread configuration | Measurements | Measurement timestamp (UTC) | Detailed report |
| --- | ---: | --- | --- |
| 1 thread | 48 | 2026-09-29T22:36:03.513Z | [Input and sampling details](configurations/julia-1-blas-1/report.md) |
| 2 threads | 48 | 2026-09-30T00:30:33.482Z | [Input and sampling details](configurations/julia-2-blas-1/report.md) |
| 4 threads | 48 | 2026-09-30T01:50:07.964Z | [Input and sampling details](configurations/julia-4-blas-1/report.md) |

Each sample is a complete double sweep. Samples continue the same state, and each CBE algorithm continues its two-site partner's state and environment.


## Hubbard / U1U1 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 11.827 seconds | 7.7272 seconds | 6.5719 seconds |
| 256 | 21.021 seconds | 12.466 seconds | 10.848 seconds |
| 512 | 61.21 seconds | 36.087 seconds | 30.252 seconds |

## Hubbard / U1U1 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 9.949 seconds | 6.2043 seconds | 5.5042 seconds |
| 512 | 30.49 seconds | 17.889 seconds | 15.903 seconds |
| 1024 | 165.41 seconds | 97.511 seconds | 84.232 seconds |

## Hubbard / U1U1 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 151.18 seconds | 96.902 seconds | 87.985 seconds |
| 256 | 320.84 seconds | 196.49 seconds | 170.96 seconds |
| 512 | 833.5 seconds | 487.7 seconds | 423.36 seconds |

## Hubbard / U1U1 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 26.032 seconds | 15.786 seconds | 13.685 seconds |
| 512 | 76.103 seconds | 45.673 seconds | 39.509 seconds |
| 1024 | 322.27 seconds | 189.51 seconds | 161.82 seconds |

## Hubbard / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 14.387 seconds | 9.8906 seconds | 10.169 seconds |
| 256 | 16.771 seconds | 11.593 seconds | 11.808 seconds |
| 512 | 24.102 seconds | 16.176 seconds | 16.174 seconds |

## Hubbard / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 7.6204 seconds | 5.6039 seconds | 5.8818 seconds |
| 512 | 11.482 seconds | 8.2517 seconds | 8.3925 seconds |
| 1024 | 30.922 seconds | 19.996 seconds | 18.032 seconds |

## Hubbard / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 146.85 seconds | 102.89 seconds | 103.44 seconds |
| 256 | 383.01 seconds | 286.02 seconds | 305.55 seconds |
| 512 | 543.72 seconds | 385.98 seconds | 410.27 seconds |

## Hubbard / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 30.998 seconds | 23.967 seconds | 24.407 seconds |
| 512 | 44.911 seconds | 30.892 seconds | 31.356 seconds |
| 1024 | 102.93 seconds | 65.272 seconds | 62.653 seconds |

## Hubbard / Z2SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 3.607 seconds | 2.3603 seconds | 2.2071 seconds |
| 256 | 7.2065 seconds | 4.3663 seconds | 3.9324 seconds |
| 512 | 23.79 seconds | 13.631 seconds | 11.933 seconds |

## Hubbard / Z2SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 4.1677 seconds | 2.7434 seconds | 2.5396 seconds |
| 512 | 15.371 seconds | 8.865 seconds | 8.1217 seconds |
| 1024 | 84.333 seconds | 49.937 seconds | 44.566 seconds |

## Hubbard / Z2SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 37.189 seconds | 23.121 seconds | 21.053 seconds |
| 256 | 130.81 seconds | 82.628 seconds | 75.223 seconds |
| 512 | 465.41 seconds | 284.09 seconds | 249.91 seconds |

## Hubbard / Z2SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 13.405 seconds | 8.3292 seconds | 7.7305 seconds |
| 512 | 49.418 seconds | 29.551 seconds | 26.073 seconds |
| 1024 | 255.7 seconds | 150.07 seconds | 130.92 seconds |

## t-t′-J-J′ / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 8.4531 seconds | 6.6434 seconds | 6.6374 seconds |
| 256 | 11.053 seconds | 7.9702 seconds | 7.553 seconds |
| 512 | 15.774 seconds | 10.65 seconds | 10.115 seconds |

## t-t′-J-J′ / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 6.3015 seconds | 5.2093 seconds | 5.0963 seconds |
| 512 | 10.076 seconds | 7.0221 seconds | 6.5966 seconds |
| 1024 | 26.05 seconds | 15.707 seconds | 14.013 seconds |

## t-t′-J-J′ / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 106.87 seconds | 76.511 seconds | 75.26 seconds |
| 256 | 131.81 seconds | 93.122 seconds | 90.853 seconds |
| 512 | 212.04 seconds | 144.55 seconds | 143.03 seconds |

## t-t′-J-J′ / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 21.542 seconds | 15.71 seconds | 15.333 seconds |
| 512 | 33.315 seconds | 22.878 seconds | 21.05 seconds |
| 1024 | 78.188 seconds | 49.849 seconds | 46.069 seconds |
