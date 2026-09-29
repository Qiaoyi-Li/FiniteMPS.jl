# Performance report

Source commit: `9c9f59b69914e9096191d9295449d06d7fee2f44`

Julia uses 1, 2, or 4 compute threads; the linear algebra backend and garbage collector each use 1 thread.

| Thread configuration | Measurements | Measurement timestamp (UTC) | Detailed report |
| --- | ---: | --- | --- |
| 1 thread | 48 | 2026-09-29T16:21:36.270Z | [Input and sampling details](configurations/julia-1-blas-1/report.md) |
| 2 threads | 48 | 2026-09-29T18:12:57.224Z | [Input and sampling details](configurations/julia-2-blas-1/report.md) |
| 4 threads | 48 | 2026-09-29T19:43:49.850Z | [Input and sampling details](configurations/julia-4-blas-1/report.md) |

Each sample is a complete double sweep. Samples continue the same state, and each CBE algorithm continues its two-site partner's state and environment.


## Hubbard / U1U1 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 9.2695 seconds | 7.1434 seconds | 6.0656 seconds |
| 128 | 13.857 seconds | 9.0965 seconds | 7.8834 seconds |
| 256 | 23.701 seconds | 15.182 seconds | 12.455 seconds |

## Hubbard / U1U1 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 3.0351 seconds | 2.3938 seconds | 2.0831 seconds |
| 128 | 4.6991 seconds | 2.9875 seconds | 2.5523 seconds |
| 256 | 8.0547 seconds | 5.367 seconds | 4.3878 seconds |

## Hubbard / U1U1 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 123.67 seconds | 84.354 seconds | 76.353 seconds |
| 128 | 191.53 seconds | 135.17 seconds | 117.57 seconds |
| 256 | 399.72 seconds | 258.33 seconds | 218.45 seconds |

## Hubbard / U1U1 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 7.235 seconds | 5.3857 seconds | 5.2212 seconds |
| 128 | 11.384 seconds | 7.3552 seconds | 6.823 seconds |
| 256 | 23.493 seconds | 15.872 seconds | 12.822 seconds |

## Hubbard / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 9.0402 seconds | 6.5942 seconds | 6.6181 seconds |
| 128 | 12.444 seconds | 9.1745 seconds | 8.6061 seconds |
| 256 | 16.826 seconds | 11.989 seconds | 11.426 seconds |

## Hubbard / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 3.1323 seconds | 3.0332 seconds | 3.0599 seconds |
| 128 | 4.8898 seconds | 3.9063 seconds | 3.6075 seconds |
| 256 | 6.2561 seconds | 4.8761 seconds | 4.9027 seconds |

## Hubbard / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 97.239 seconds | 61.482 seconds | 57.142 seconds |
| 128 | 206.01 seconds | 147.53 seconds | 143.92 seconds |
| 256 | 333.89 seconds | 244.61 seconds | 242.15 seconds |

## Hubbard / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 11.748 seconds | 8.6973 seconds | 8.775 seconds |
| 128 | 17.46 seconds | 13.58 seconds | 13.175 seconds |
| 256 | 20.962 seconds | 17.763 seconds | 18.254 seconds |

## Hubbard / Z2SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 2.0601 seconds | 1.5743 seconds | 1.4786 seconds |
| 128 | 3.3172 seconds | 2.6876 seconds | 2.3602 seconds |
| 256 | 6.1201 seconds | 4.6904 seconds | 4.4781 seconds |

## Hubbard / Z2SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 849.04 milliseconds | 751.83 milliseconds | 831.6 milliseconds |
| 128 | 1.2056 seconds | 1.058 seconds | 1.1305 seconds |
| 256 | 2.2876 seconds | 1.828 seconds | 1.6992 seconds |

## Hubbard / Z2SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 14.769 seconds | 10.79 seconds | 9.252 seconds |
| 128 | 34.187 seconds | 26.143 seconds | 22.307 seconds |
| 256 | 116.49 seconds | 84.089 seconds | 71.742 seconds |

## Hubbard / Z2SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 1.6813 seconds | 1.4023 seconds | 1.3632 seconds |
| 128 | 3.1645 seconds | 2.5189 seconds | 2.3056 seconds |
| 256 | 9.3171 seconds | 7.048 seconds | 5.6753 seconds |

## t-t′-J-J′ / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 5.5339 seconds | 4.2983 seconds | 3.9315 seconds |
| 128 | 8.8031 seconds | 5.5633 seconds | 5.0403 seconds |
| 256 | 9.6204 seconds | 7.6553 seconds | 6.9013 seconds |

## t-t′-J-J′ / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 2.6505 seconds | 2.4055 seconds | 2.162 seconds |
| 128 | 3.5361 seconds | 3.0854 seconds | 2.8545 seconds |
| 256 | 5.0143 seconds | 3.9843 seconds | 3.5135 seconds |

## t-t′-J-J′ / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 63.937 seconds | 50.023 seconds | 44.69 seconds |
| 128 | 106.37 seconds | 73.615 seconds | 75.74 seconds |
| 256 | 116.08 seconds | 80.579 seconds | 79.201 seconds |

## t-t′-J-J′ / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 64 | 8.2889 seconds | 6.9181 seconds | 6.4835 seconds |
| 128 | 15.647 seconds | 11.833 seconds | 11.872 seconds |
| 256 | 17.087 seconds | 12.834 seconds | 14.003 seconds |
