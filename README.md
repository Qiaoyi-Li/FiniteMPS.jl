# FiniteMPS.jl

A julia package for finite MPS/MPO-based computations of ground-state, finite-temperature and dynamical properties.

[![][docs-latest-img]][docs-latest-url]

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://qiaoyi-li.github.io/FiniteMPS.jl/dev


## Features
### Versatility
- FiniteMPS.jl integrates multiple algorithms for studying a quantum many-body system: DMRG for ground state, [tanTRG](https://doi.org/10.1103/PhysRevLett.130.226502) for finite-T and [TDVP](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.107.070601) for dynamics, see [Tutorial/Heisenberg](https://qiaoyi-li.github.io/FiniteMPS.jl/dev/tutorial/heisenberg).
- Both spin and fermion systems are supported, see [Tutorial/Hubbard](https://qiaoyi-li.github.io/FiniteMPS.jl/dev/tutorial/hubbard).

### Convenience
- The MPS-level operations are separated with the lower tensor-level operations so that modifying codes with different symmetries, lattices and models is quite simple.
- We provide an interface to adding arbitrary multi-site interactions so that the Hamiltonian MPO can be generated in a simple and general way, see [Tutorial/Hamiltonian](https://qiaoyi-li.github.io/FiniteMPS.jl/dev/tutorial/hamiltonian).
- We also provide a similar interface for calculating arbitrary observables (e.g. multi-site correlations), see [Tutorial/Observable](https://qiaoyi-li.github.io/FiniteMPS.jl/dev/tutorial/observable). 

### Performance
- We use the state-of-the-art tensor library [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) to perform the basic tensor operations, so that non-abelian symmetries can significantly accelerate computations.
- The hamiltonian MPO can be represented as a sparse one with abstract horizontal bonds so that the interaction terms can be distributed to different threads or workers, together with nested multi-threaded BLAS (with the help of [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl)).
- [Controlled bond expansion (CBE)](https://doi.org/10.1103/PhysRevLett.130.246402) techniques are implemented (modified for compatibility with parallelism) to reduce the complexity, e.g. $O(D^3d^2)$ to $O(D^3d)$ for DMRG and TDVP.

## Quick start
FiniteMPS.jl is a registered package. To install it, just press "]" in julia REPL and type
```julia
pkg> add FiniteMPS
```
 
A short tutorial to use this package is provided in [documentation](https://qiaoyi-li.github.io/FiniteMPS.jl/dev/), in where we give some demo scripts that solve ground-state, finite-T and dynamical properties of concrete models. 

If you encounter any problem, please feel free to contact us (liqiaoyi@itp.ac.cn) or submit an issue.

### Local space
The first step is to choose a local space, i.e. the local 1-site Hilbert space and some 1-site operators according to the model and symmetry used. We predefine some in folder `LocalSpace` and you can use the documentation to see which operators are provided in it, e.g 
```julia 
help?> U₁Spin
```
> [!NOTE] Note:
It is impossible to predefine all local spaces so you must write a new one if that you need is not in it. Moreover, some `no method` errors may occur when using a new defined one due to the multiple-dispatch-based implementations may not cover all kinds of operators. Please feel free to submit an issue if this happens and I will deal with it as soon as possible.

### Hamiltonian
After that, you need to tell the program the interactions in the hamiltonian. We use a tree-like struct `InteractionTree` to store them and provide an interface named `addIntr!` to add terms. The following is a demo code to generate the hamiltonian of an AFM Heisenberg chain.
```julia
# generate hamiltonian MPO
L = 32
Root = InteractionTreeNode()
for i in 1:L-1
     # SzSz
     addIntr!(Root,
          (U₁Spin.Sz, U₁Spin.Sz),
          (i, i+1),
          1.0;
          name = (:Sz, :Sz))
     # (S+S- + h.c.)/2
     addIntr!(Root, 
          U₁Spin.S₊₋,
          (i, i+1),
          0.5;
          name = (:Sp, :Sm))
     addIntr!(Root,
          U₁Spin.S₋₊,
          (i, i+1),
          0.5;
          name = (:Sm, :Sp))
end
Ham = AutomataMPO(Root)
```

### Main script
Although the `FiniteMPS` package is written in an object-oriented style, a procedure-oriented script is still required for a concrete question. You can freely design it with the following sweep methods such as 
- DMRGSweep2!, DMRGSweep1!\
   The standard 2-site/1-site DMRG sweep.
- TDVPSweep2!, TDVPSweep1!\
   The standard 2-site/1-site TDVP sweep.

being the usually used building blocks.

Note a trilayer tensor network (namely environment), e.g. $\langle \Psi|H|\Psi\rangle$ where $\ket{\Psi}$ the MPS and $H$ the hamiltonian MPO, instead of the state $\ket\Psi$, should be passed in. The following code will initialize a random MPS and then generate the environment.
```julia
# initialize
Ψ = randMPS(L, U₁Spin.pspace, Rep[U₁](i => 1 for i in -1:1//2:1))
Env = Environment(Ψ', Ham, Ψ)
```
More detailed usages of the above functions can be found in the documentation.

> [!NOTE] Note: 
Symmetries may limit the choice of auxiliary space, e.g. the spin quantum number must be integers or half integers alternately along the sites for spin-1/2 systems as fusing the physical space will exactly shift quantum number by 1/2. If the auxiliary space (`Rep[U₁](i => 1 for i in -1:1//2:1)` here) is not chosen appropriately to be compatible with the symmetry, some errors will occur when generating the MPS or later.

For example, you can use `DMRGSweep2!` to obtain the ground state MPS
```julia
# 2-DMRG sweeps
D = 256
for i in 1:5
     t = @elapsed info, timer = DMRGSweep2!(Env; trunc = truncdim(D))
     println("Sweep $(i), En = $(info[2][1].Eg), wall time = $(t)s")
end
```

### Observables
Then, you can use a similar interface `addObs!` to generate the tree that stores the observables to be calculated.
```julia
# calculate observables
Tree = ObservableTree()
# all to all spin correlation
for i in 1:L, j in i + 1:L
     addObs!(Tree, (U₁Spin.Sz, U₁Spin.Sz), (i, j), (false, false); name = (:Sz, :Sz))
end
# local moment
for i in 1:L
     addObs!(Tree, U₁Spin.Sz, i; name = :Sz)
end

calObs!(Tree, Ψ)
# collect the results from Tree
Obs = convert(NamedTuple, Tree)
```

Now you can obtain the spin correlation $\langle S_i^z S_j^z\rangle$ via `Obs.SzSz[(i,j)]` and local moment $\langle S_i^z \rangle$ via `Obs.Sz[(i,)]` (should be zero up to a noise due to the $SU_2$ symmetry of the Heisenberg model).

> [!WARNING] Warning:
  There may exist undiscovered bugs, therefore, please use it at your own risk. Benchmark with exact diagonalization (ED) is strongly recommended before using it to simulate any new models.

## 
