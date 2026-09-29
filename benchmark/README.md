# Sweep performance benchmarks

Run `julia --project=benchmark benchmark/collect.jl` to collect the 1, 2 and
4 Julia-thread configurations, with one BLAS and one GC thread. The report
contains 16 tables: four models × four algorithms, with D=64/128/256 as rows
and thread counts as columns. Each sample times a complete left-to-right and
right-to-left sweep. The default is at most three samples, `evals=1`, a
600-second budget per case, and one smallest-D warmup per model/algorithm.
The budget accommodates the larger two-site TDVP cases.

All models use a 32-site YC4×8 square cylinder with snake ordering, t=1 and
t′=-0.2. Hubbard uses U=8 with U1U1, U1SU2 or Z2SU2 symmetry. The projected
tJ model uses U1SU2 and J=0.5, J′=0.02, including the −J ni nj/4 terms.
The three U1 ground-state profiles have N=28 and zero spin projection/total
spin. The Z2 ground-state profile is an even singlet with μ=2. All four thermal
profiles are grand canonical at μ=2 and T/t=1.

`presets/*.json` are repository input data for local runs and cloud CI.
They contain every bond's sector multiplicities and the parameters of
the short offline calculation that produced them. There is no offline generator,
saved numerical MPS/MPO, or cooling run in this repository or in the benchmark.
The profiles are approximate workload shapes, not converged physical results.

Inputs are Float64 random block tensors reconstructed from those profiles with
seed 20260929, then canonicalized and normalized before timing. Each model/D
constructs one MPS and environment for 2-DMRG followed by CBE-DMRG, and one MPO
and environment for 2-TDVP followed by CBE-TDVP. Warmup and measured samples
continue these mutable states without resetting them. The two algorithms are
timed separately; CBE consequently starts from its two-site partner's output.

Every sweep uses K=8, `truncrank(D)`, `GCstep=false` and `GCsweep=false`.
CBE-DMRG uses `NaiveCBE(2D,1e-8;rsvd=true)`; CBE-TDVP uses
`NaiveCBE(D+div(D,8),1e-8;rsvd=true)`. Timed TDVP uses dt=-0.1 for the complete
symmetric step (−0.05 in each direction). Other algorithm options retain their
defaults. Automatic Julia GC remains enabled; the harness collects after each
case, outside the measured operation.

Offline ground profiles used two 2-DMRG double sweeps per D with continuation
64→128→256. Each thermal profile used one CBE-TDVP cooling run from identity:
first δβ=2^-10, then δβ=β, reaching β=1 in 11 double sweeps. Since the stored
operator represents exp(−βH/2), cooling used dt=−δβ/2. Only the resulting
sector configurations and their provenance are retained here: 24 profiles from
24 ground-state sweeps and 12 cooling runs (132 thermal sweeps).
