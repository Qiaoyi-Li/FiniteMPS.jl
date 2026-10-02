# Performance report

Source commit: `74ab3410e8fe4584e9d12c8ebd4b870397b410b1`

Julia uses 1, 2, or 4 compute threads; the linear algebra backend and garbage collector each use 1 thread.

| Thread configuration | Measurements | Measurement timestamp (UTC) | Detailed report |
| --- | ---: | --- | --- |
| 1 thread | 48 | 2026-10-02T13:28:58.079Z | [Input and sampling details](configurations/julia-1-blas-1/report.md) |
| 2 threads | 48 | 2026-10-02T15:21:22.066Z | [Input and sampling details](configurations/julia-2-blas-1/report.md) |
| 4 threads | 48 | 2026-10-02T16:40:33.770Z | [Input and sampling details](configurations/julia-4-blas-1/report.md) |

Each sample is a complete double sweep. Samples continue the same state, and each CBE algorithm continues its two-site partner's state and environment.


## Hubbard / U1U1 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 11.963 seconds | 7.2864 seconds | 6.6901 seconds |
| 256 | 21.046 seconds | 12.157 seconds | 10.796 seconds |
| 512 | 63.822 seconds | 35.432 seconds | 30.544 seconds |

## Hubbard / U1U1 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 9.9783 seconds | 6.4034 seconds | 5.5989 seconds |
| 512 | 29.598 seconds | 17.434 seconds | 15.928 seconds |
| 1024 | 163.03 seconds | 93.737 seconds | 82.769 seconds |

## Hubbard / U1U1 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 151.46 seconds | 93.205 seconds | 85.203 seconds |
| 256 | 308.42 seconds | 188.56 seconds | 168.23 seconds |
| 512 | 817.91 seconds | 486.53 seconds | 433.28 seconds |

## Hubbard / U1U1 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 25.308 seconds | 15.245 seconds | 13.393 seconds |
| 512 | 73.606 seconds | 44.271 seconds | 39.431 seconds |
| 1024 | 314.96 seconds | 186.65 seconds | 163.02 seconds |

## Hubbard / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 13.557 seconds | 9.5186 seconds | 10.079 seconds |
| 256 | 16.29 seconds | 11.222 seconds | 11.724 seconds |
| 512 | 23.301 seconds | 15.738 seconds | 15.93 seconds |

## Hubbard / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 7.233 seconds | 5.5236 seconds | 5.9101 seconds |
| 512 | 11.118 seconds | 8.0453 seconds | 8.2298 seconds |
| 1024 | 30.227 seconds | 19.585 seconds | 17.735 seconds |

## Hubbard / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 141.85 seconds | 97.494 seconds | 102.33 seconds |
| 256 | 367.09 seconds | 270.05 seconds | 304.77 seconds |
| 512 | 504.85 seconds | 369.64 seconds | 404.79 seconds |

## Hubbard / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 30.718 seconds | 23.715 seconds | 25.227 seconds |
| 512 | 42.085 seconds | 29.91 seconds | 31.687 seconds |
| 1024 | 95.463 seconds | 64.576 seconds | 62.601 seconds |

## Hubbard / Z2SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 3.631 seconds | 2.4001 seconds | 2.3043 seconds |
| 256 | 7.0733 seconds | 4.5784 seconds | 4.1136 seconds |
| 512 | 24.06 seconds | 13.817 seconds | 12.073 seconds |

## Hubbard / Z2SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 4.1984 seconds | 2.7448 seconds | 2.6742 seconds |
| 512 | 13.853 seconds | 8.7419 seconds | 8.1462 seconds |
| 1024 | 80.33 seconds | 47.525 seconds | 43.773 seconds |

## Hubbard / Z2SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 36.892 seconds | 23.762 seconds | 21.396 seconds |
| 256 | 125.27 seconds | 81.255 seconds | 77.536 seconds |
| 512 | 467.22 seconds | 280.72 seconds | 258.48 seconds |

## Hubbard / Z2SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 12.825 seconds | 8.3687 seconds | 7.7867 seconds |
| 512 | 47.223 seconds | 29.595 seconds | 26.374 seconds |
| 1024 | 246.71 seconds | 145.78 seconds | 130.39 seconds |

## t-t′-J-J′ / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 8.4672 seconds | 6.5738 seconds | 6.9082 seconds |
| 256 | 11.026 seconds | 7.8524 seconds | 7.8267 seconds |
| 512 | 15.829 seconds | 10.432 seconds | 10.342 seconds |

## t-t′-J-J′ / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 6.2902 seconds | 5.1933 seconds | 5.3118 seconds |
| 512 | 10.062 seconds | 6.8944 seconds | 6.7024 seconds |
| 1024 | 24.446 seconds | 15.129 seconds | 13.558 seconds |

## t-t′-J-J′ / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 103.41 seconds | 73.635 seconds | 75.732 seconds |
| 256 | 125.2 seconds | 90.544 seconds | 92.192 seconds |
| 512 | 201.49 seconds | 141.76 seconds | 142.28 seconds |

## t-t′-J-J′ / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 21.041 seconds | 16.077 seconds | 15.475 seconds |
| 512 | 31.444 seconds | 23.192 seconds | 21.104 seconds |
| 1024 | 74.555 seconds | 49.729 seconds | 45.874 seconds |
