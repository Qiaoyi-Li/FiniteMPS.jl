# Performance report

Source commit: `a19056f4b950167a7207ace2ca4c845c2d644393`

Julia uses 1, 2, or 4 compute threads; the linear algebra backend and garbage collector each use 1 thread.

| Thread configuration | Measurements | Measurement timestamp (UTC) | Detailed report |
| --- | ---: | --- | --- |
| 1 thread | 48 | 2026-11-05T11:32:28.754Z | [Input and sampling details](configurations/julia-1-blas-1/report.md) |
| 2 threads | 48 | 2026-11-05T13:29:40.961Z | [Input and sampling details](configurations/julia-2-blas-1/report.md) |
| 4 threads | 48 | 2026-11-05T14:52:17.159Z | [Input and sampling details](configurations/julia-4-blas-1/report.md) |

Each sample is a complete double sweep. Samples continue the same state, and each CBE algorithm continues its two-site partner's state and environment.


## Hubbard / U1U1 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 12.834 seconds | 7.4494 seconds | 6.526 seconds |
| 256 | 23.723 seconds | 12.857 seconds | 10.739 seconds |
| 512 | 62.909 seconds | 38.256 seconds | 31.222 seconds |

## Hubbard / U1U1 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 10.026 seconds | 6.4186 seconds | 5.4826 seconds |
| 512 | 29.572 seconds | 18.434 seconds | 16.03 seconds |
| 1024 | 166.46 seconds | 98.664 seconds | 86.297 seconds |

## Hubbard / U1U1 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 155.02 seconds | 95.201 seconds | 90.273 seconds |
| 256 | 319.69 seconds | 197.76 seconds | 176.8 seconds |
| 512 | 858.34 seconds | 521.56 seconds | 452.54 seconds |

## Hubbard / U1U1 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 25.957 seconds | 15.664 seconds | 13.85 seconds |
| 512 | 74.152 seconds | 45.669 seconds | 39.594 seconds |
| 1024 | 334.96 seconds | 204.8 seconds | 171.27 seconds |

## Hubbard / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 13.421 seconds | 9.8082 seconds | 10.287 seconds |
| 256 | 16.221 seconds | 11.492 seconds | 12.16 seconds |
| 512 | 23.48 seconds | 16.18 seconds | 16.666 seconds |

## Hubbard / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 7.1709 seconds | 5.6301 seconds | 5.8552 seconds |
| 512 | 11.357 seconds | 8.2148 seconds | 8.6651 seconds |
| 1024 | 29.988 seconds | 19.768 seconds | 17.987 seconds |

## Hubbard / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 149.28 seconds | 103.96 seconds | 103.62 seconds |
| 256 | 376.02 seconds | 283.7 seconds | 312.88 seconds |
| 512 | 525.68 seconds | 388.82 seconds | 422.74 seconds |

## Hubbard / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 30.986 seconds | 24.136 seconds | 25.737 seconds |
| 512 | 44.325 seconds | 31.603 seconds | 32.056 seconds |
| 1024 | 98.022 seconds | 67.222 seconds | 62.69 seconds |

## Hubbard / Z2SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 3.8535 seconds | 2.4201 seconds | 2.1823 seconds |
| 256 | 7.5814 seconds | 4.6387 seconds | 4.0812 seconds |
| 512 | 24.744 seconds | 14.506 seconds | 12.377 seconds |

## Hubbard / Z2SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 4.5263 seconds | 2.894 seconds | 2.6009 seconds |
| 512 | 14.359 seconds | 8.9285 seconds | 8.2318 seconds |
| 1024 | 82.575 seconds | 48.956 seconds | 43.417 seconds |

## Hubbard / Z2SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 36.858 seconds | 24.034 seconds | 21.744 seconds |
| 256 | 130.93 seconds | 84.427 seconds | 77.902 seconds |
| 512 | 476.15 seconds | 299.06 seconds | 262.72 seconds |

## Hubbard / Z2SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 13.501 seconds | 8.4831 seconds | 7.7435 seconds |
| 512 | 49.976 seconds | 30.297 seconds | 26.868 seconds |
| 1024 | 254.02 seconds | 153.22 seconds | 133.11 seconds |

## t-t′-J-J′ / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 8.643 seconds | 6.7021 seconds | 6.945 seconds |
| 256 | 11.81 seconds | 8.0759 seconds | 7.9759 seconds |
| 512 | 16.433 seconds | 10.773 seconds | 10.491 seconds |

## t-t′-J-J′ / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 6.6977 seconds | 5.3138 seconds | 5.3525 seconds |
| 512 | 10.439 seconds | 7.184 seconds | 6.769 seconds |
| 1024 | 26.136 seconds | 15.994 seconds | 13.525 seconds |

## t-t′-J-J′ / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 110.08 seconds | 75.247 seconds | 78.719 seconds |
| 256 | 137.97 seconds | 91.198 seconds | 95.076 seconds |
| 512 | 217.24 seconds | 143.14 seconds | 148.04 seconds |

## t-t′-J-J′ / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 22.881 seconds | 15.674 seconds | 15.917 seconds |
| 512 | 34.205 seconds | 22.621 seconds | 21.688 seconds |
| 1024 | 78.371 seconds | 49.657 seconds | 48.167 seconds |
