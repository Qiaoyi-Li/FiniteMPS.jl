# Performance report

Source commit: `a5d972830d50d43a5ddfa58242ee44556cad2b9e`

Julia uses 1, 2, or 4 compute threads; the linear algebra backend and garbage collector each use 1 thread.

| Thread configuration | Measurements | Measurement timestamp (UTC) | Detailed report |
| --- | ---: | --- | --- |
| 1 thread | 48 | 2026-10-02T11:49:08.738Z | [Input and sampling details](configurations/julia-1-blas-1/report.md) |
| 2 threads | 48 | 2026-10-02T13:38:24.954Z | [Input and sampling details](configurations/julia-2-blas-1/report.md) |
| 4 threads | 48 | 2026-10-02T14:56:17.274Z | [Input and sampling details](configurations/julia-4-blas-1/report.md) |

Each sample is a complete double sweep. Samples continue the same state, and each CBE algorithm continues its two-site partner's state and environment.


## Hubbard / U1U1 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 11.249 seconds | 7.1698 seconds | 6.3711 seconds |
| 256 | 19.833 seconds | 12.138 seconds | 10.339 seconds |
| 512 | 58.259 seconds | 35.566 seconds | 29.731 seconds |

## Hubbard / U1U1 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 9.4283 seconds | 6.2368 seconds | 5.457 seconds |
| 512 | 28.163 seconds | 17.165 seconds | 15.467 seconds |
| 1024 | 152.5 seconds | 92.28 seconds | 81.123 seconds |

## Hubbard / U1U1 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 143.54 seconds | 95.538 seconds | 85.844 seconds |
| 256 | 296.75 seconds | 190.73 seconds | 168.29 seconds |
| 512 | 787.74 seconds | 475.84 seconds | 415.75 seconds |

## Hubbard / U1U1 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 24.669 seconds | 15.134 seconds | 13.146 seconds |
| 512 | 73.528 seconds | 43.337 seconds | 37.878 seconds |
| 1024 | 309.31 seconds | 182.75 seconds | 157.78 seconds |

## Hubbard / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 13.109 seconds | 9.0388 seconds | 9.7441 seconds |
| 256 | 15.637 seconds | 11.209 seconds | 11.46 seconds |
| 512 | 22.46 seconds | 15.356 seconds | 15.564 seconds |

## Hubbard / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 6.8582 seconds | 5.309 seconds | 5.6806 seconds |
| 512 | 10.654 seconds | 7.8311 seconds | 8.1512 seconds |
| 1024 | 29.298 seconds | 18.874 seconds | 17.288 seconds |

## Hubbard / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 137.98 seconds | 95.917 seconds | 99.851 seconds |
| 256 | 352.82 seconds | 265.45 seconds | 296.74 seconds |
| 512 | 492.31 seconds | 361.93 seconds | 393.94 seconds |

## Hubbard / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 28.898 seconds | 22.851 seconds | 24.512 seconds |
| 512 | 40.764 seconds | 29.009 seconds | 30.237 seconds |
| 1024 | 92.723 seconds | 61.291 seconds | 60.226 seconds |

## Hubbard / Z2SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 3.6339 seconds | 2.4091 seconds | 2.2451 seconds |
| 256 | 7.2176 seconds | 4.461 seconds | 3.9775 seconds |
| 512 | 23.29 seconds | 13.388 seconds | 11.641 seconds |

## Hubbard / Z2SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 4.0399 seconds | 2.6613 seconds | 2.5538 seconds |
| 512 | 14.19 seconds | 8.5501 seconds | 7.8521 seconds |
| 1024 | 81.76 seconds | 46.676 seconds | 42.471 seconds |

## Hubbard / Z2SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 35.902 seconds | 23.647 seconds | 20.63 seconds |
| 256 | 123.66 seconds | 80.615 seconds | 73.121 seconds |
| 512 | 449.25 seconds | 279.57 seconds | 247.7 seconds |

## Hubbard / Z2SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 12.668 seconds | 8.2917 seconds | 7.5026 seconds |
| 512 | 49.051 seconds | 28.637 seconds | 25.601 seconds |
| 1024 | 237.86 seconds | 142.56 seconds | 126.01 seconds |

## t-t′-J-J′ / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 8.0179 seconds | 6.3085 seconds | 6.4921 seconds |
| 256 | 10.757 seconds | 7.7699 seconds | 7.4082 seconds |
| 512 | 14.724 seconds | 10.227 seconds | 9.9251 seconds |

## t-t′-J-J′ / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 5.9684 seconds | 5.1845 seconds | 4.9472 seconds |
| 512 | 9.7292 seconds | 6.7852 seconds | 6.4051 seconds |
| 1024 | 24.249 seconds | 14.925 seconds | 13.456 seconds |

## t-t′-J-J′ / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 98.307 seconds | 71.179 seconds | 73.178 seconds |
| 256 | 123.34 seconds | 87.194 seconds | 89.836 seconds |
| 512 | 202.08 seconds | 136.31 seconds | 140.23 seconds |

## t-t′-J-J′ / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 20.287 seconds | 14.94 seconds | 14.853 seconds |
| 512 | 30.855 seconds | 21.631 seconds | 20.58 seconds |
| 1024 | 72.963 seconds | 46.758 seconds | 45.467 seconds |
