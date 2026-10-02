# Performance report

Source commit: `74ab3410e8fe4584e9d12c8ebd4b870397b410b1`

Julia uses 1, 2, or 4 compute threads; the linear algebra backend and garbage collector each use 1 thread.

| Thread configuration | Measurements | Measurement timestamp (UTC) | Detailed report |
| --- | ---: | --- | --- |
| 1 thread | 48 | 2026-10-02T12:10:07.646Z | [Input and sampling details](configurations/julia-1-blas-1/report.md) |
| 2 threads | 48 | 2026-10-02T14:10:38.060Z | [Input and sampling details](configurations/julia-2-blas-1/report.md) |
| 4 threads | 48 | 2026-10-02T15:33:40.106Z | [Input and sampling details](configurations/julia-4-blas-1/report.md) |

Each sample is a complete double sweep. Samples continue the same state, and each CBE algorithm continues its two-site partner's state and environment.


## Hubbard / U1U1 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 13.279 seconds | 8.6251 seconds | 7.7512 seconds |
| 256 | 22.263 seconds | 13.786 seconds | 12.166 seconds |
| 512 | 67.613 seconds | 40.847 seconds | 33.737 seconds |

## Hubbard / U1U1 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 10.262 seconds | 6.9876 seconds | 6.1329 seconds |
| 512 | 30.733 seconds | 18.936 seconds | 17.414 seconds |
| 1024 | 162.39 seconds | 98.541 seconds | 86.229 seconds |

## Hubbard / U1U1 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 167.17 seconds | 102.3 seconds | 94.418 seconds |
| 256 | 340.4 seconds | 207.51 seconds | 184.95 seconds |
| 512 | 885.96 seconds | 524.24 seconds | 477.12 seconds |

## Hubbard / U1U1 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 28.855 seconds | 17.499 seconds | 15.509 seconds |
| 512 | 86.88 seconds | 51.578 seconds | 43.055 seconds |
| 1024 | 330.73 seconds | 195.41 seconds | 173.44 seconds |

## Hubbard / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 14.532 seconds | 10.144 seconds | 10.91 seconds |
| 256 | 17.205 seconds | 11.883 seconds | 12.788 seconds |
| 512 | 25.77 seconds | 16.912 seconds | 17.489 seconds |

## Hubbard / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 8.1336 seconds | 6.3607 seconds | 6.7814 seconds |
| 512 | 12.21 seconds | 8.682 seconds | 9.2893 seconds |
| 1024 | 33.516 seconds | 21.051 seconds | 19.633 seconds |

## Hubbard / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 154.89 seconds | 105.12 seconds | 108.71 seconds |
| 256 | 383.33 seconds | 280.52 seconds | 309.91 seconds |
| 512 | 543.36 seconds | 387.93 seconds | 412 seconds |

## Hubbard / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 32.59 seconds | 24.429 seconds | 25.072 seconds |
| 512 | 45.801 seconds | 32.159 seconds | 32.604 seconds |
| 1024 | 105.56 seconds | 67.785 seconds | 65.325 seconds |

## Hubbard / Z2SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 4.4379 seconds | 2.8048 seconds | 2.5689 seconds |
| 256 | 8.2565 seconds | 4.8125 seconds | 4.4283 seconds |
| 512 | 27.074 seconds | 15.443 seconds | 13.638 seconds |

## Hubbard / Z2SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 4.5315 seconds | 3.1339 seconds | 2.9263 seconds |
| 512 | 15.342 seconds | 9.5031 seconds | 8.5044 seconds |
| 1024 | 80.384 seconds | 47.769 seconds | 44.175 seconds |

## Hubbard / Z2SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 41.864 seconds | 26.684 seconds | 23.441 seconds |
| 256 | 142.36 seconds | 88.573 seconds | 83.107 seconds |
| 512 | 523.8 seconds | 314.78 seconds | 290.37 seconds |

## Hubbard / Z2SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 14.384 seconds | 9.2489 seconds | 8.7467 seconds |
| 512 | 53.539 seconds | 31.782 seconds | 27.794 seconds |
| 1024 | 251.53 seconds | 145.19 seconds | 132.09 seconds |

## t-t′-J-J′ / U1SU2 · 2-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 9.3985 seconds | 7.3069 seconds | 7.5386 seconds |
| 256 | 12.322 seconds | 8.4075 seconds | 8.5379 seconds |
| 512 | 17.596 seconds | 11.177 seconds | 11.082 seconds |

## t-t′-J-J′ / U1SU2 · CBE-DMRG

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 7.33 seconds | 5.8264 seconds | 5.9597 seconds |
| 512 | 11.355 seconds | 7.7334 seconds | 7.3576 seconds |
| 1024 | 28.068 seconds | 16.832 seconds | 15.081 seconds |

## t-t′-J-J′ / U1SU2 · 2-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 128 | 112.12 seconds | 77.091 seconds | 82.947 seconds |
| 256 | 137.54 seconds | 92.059 seconds | 99.562 seconds |
| 512 | 222.89 seconds | 149.71 seconds | 153.92 seconds |

## t-t′-J-J′ / U1SU2 · CBE-TDVP

| D | 1 thread | 2 threads | 4 threads |
| --- | ---: | ---: | ---: |
| 256 | 23.227 seconds | 16.444 seconds | 16.854 seconds |
| 512 | 33.149 seconds | 23.339 seconds | 22.351 seconds |
| 1024 | 82.52 seconds | 51.019 seconds | 49.634 seconds |
