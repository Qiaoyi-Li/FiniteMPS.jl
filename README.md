# FiniteMPS.jl

A julia package for finite MPS/MPO-based computations of ground-state, finite-temperature and dynamical properties.

[![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://qiaoyi-li.github.io/FiniteMPS.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://qiaoyi-li.github.io/FiniteMPS.jl/dev

## Features
### Versatility
- FiniteMPS.jl integrates multiple algorithms for studying a quantum many-body system: DMRG for ground state, [tanTRG](https://doi.org/10.1103/PhysRevLett.130.226502) for finite-T and [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601) for dynamics, see [Tutorial/Heisenberg](https://qiaoyi-li.github.io/FiniteMPS.jl/stable/tutorial/Heisenberg).
- Both spin and fermion systems are supported, see [Tutorial/Hubbard](https://qiaoyi-li.github.io/FiniteMPS.jl/stable/tutorial/Hubbard).

### Convenience
- The MPS-level operations are separated with the lower tensor-level operations so that modifying codes with different symmetries, lattices and models is quite simple.
- We provide an interface to adding arbitrary multi-site interactions so that the Hamiltonian MPO can be generated in a simple and general way, see [Tutorial/Hamiltonian](https://qiaoyi-li.github.io/FiniteMPS.jl/stable/tutorial/Hamiltonian).
- We also provide a similar interface for calculating arbitrary observables (e.g. multi-site correlations), see [Tutorial/Observable](https://qiaoyi-li.github.io/FiniteMPS.jl/stable/tutorial/Observable). 

### Performance
- We use the state-of-the-art tensor library [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) to perform the basic tensor operations, so that non-abelian symmetries can significantly accelerate computations.
- The hamiltonian MPO can be represented as a sparse one with abstract horizontal bonds so that the interaction terms can be distributed to different threads or workers, together with nested multi-threaded BLAS (with the help of [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl)).
- [Controlled bond expansion (CBE)](https://doi.org/10.1103/PhysRevLett.130.246402) techniques are implemented (modified for compatibility with parallelism) to reduce the complexity, e.g. $O(D^3d^2)$ to $O(D^3d)$ for DMRG and TDVP.

## Quick start
FiniteMPS.jl is a registered package. To install it, just press "]" in julia REPL and type
```julia
pkg> add FiniteMPS
```
 
A short tutorial to use this package is provided in [documentation](https://qiaoyi-li.github.io/FiniteMPS.jl/stable/), in where we give some demo scripts that solve ground-state, finite-T and dynamical properties of concrete models. 

If you encounter any problem, please feel free to contact us (liqiaoyi@itp.ac.cn) or submit an issue.