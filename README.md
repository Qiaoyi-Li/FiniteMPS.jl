# FiniteMPS.jl

This package provides the basic MPS operations and also some high-level algorithms. 

## Features
### Performance
I write this package aim to study some heavy quantum many-body problems (e.g. high-$T_c$ superconductivity, typical bond dimension $D \sim 10^4$  ), therefore the performance is the the most important metric.

- Non-abelian symmetries (e.g. $U_1\times SU_2$ symmetry for spin-1/2 fermions) can significantly accelerate the computation, therefore we choose [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) as the backend to perform the basic tensor operations.
- The hamiltonian MPO can be represented as a sparse one with abstract horizontal bonds so that the interaction terms can be distributed to different threads or workers, together with nested multi-threaded BLAS (with the help of [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl)).
- The local operators are classified and wrapped as parametric types for further optimizations via multiple dispatch.
- [Controlled bond expansion (CBE)](https://doi.org/10.1103/PhysRevLett.130.246402) techniques are implemented (modified for compatibility with parallelism) to reduce the complexity, e.g. $O(D^3d^2)$ to $O(D^3d)$ for DMRG and TDVP.

### Convenience

- The MPS-level operations are separated with the lower tensor-level operations so modifying codes with different symmetries, lattices and models is quite simple.
- We provide a hamiltonian generator via automata (like `OpSum` in [ITensors.jl](https://github.com/ITensor/ITensors.jl)) so that the interactions can be added in a simple and general way, details please see the tutorials.
- We also provide some similar interfaces so that you can measure the  observales and correlations conveniently. Note these computations can also be multi-threaded.
  
### MPO supports
[Our group](https://www.cqm2itp.com/) works on purification-based finite-temperature simulations therefore we provide more MPO supports than other MPS packages.

- Almost all methods for MPS are also implemented for MPO.
- Some finite-temperature algorithms such as series-expansion thermal tensor network ([SETTN](https://doi.org/10.1103/PhysRevB.95.161104)) and tangent space tensor renormalization group ([tanTRG](https://doi.org/10.1103/PhysRevLett.130.226502)) are provided.

### Versatility
Now this package contains

- DMRG for searching ground states.
- SETTN and tanTRG for finite-temperature simulations. Both thermal quantites and observables can be obtained.
- Ground state and finite-temperature dynamics can be studied with real-time evolution based on TDVP. Note this is still under development.

## Tutorial

### Installation
To install this package, you can press "]" in REPL and type
```julia
pkg> add FiniteMPS
```

### Multi-threading settings
- Only multi-threading is stable in current version, please make sure Julia is started with [multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading). It will throw a warning to remind you when loading `FiniteMPS` with a single thread.
  
  TODO: I will add single-thread mode and try to support multi-processing in the future.

- If the linear algebra backend is `OpenBlas`, nested multi-threading is forbidden (a related discussion [here](https://carstenbauer.github.io/ThreadPinning.jl/stable/explanations/blas/)). So just close the parallelism of BLAS via
  ```julia
  BLAS.set_num_threads(1)
  ```

- If using MKL as the linear algebra backend, you can set the number of blas threads similarly and nested multi-threading can be supported in this case. Note MKL is invalid for some cpus and you should also close the BLAS parallelism in such case. You can follow [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl) to check if MKL is loaded successfully or not.
  
   > [!WARNING] Warning: 
     Please make sure the total threads number is not larger than the cpu cores (physical, without hyper-threading). Otherwise, the performance will become much worse due to confliction. You can estimate the total threads via    
     ```julia
     Threads.nthreads() * BLAS.get_num_threads()
     ```   


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

## Contributions
### Authors
- [Qiaoyi Li](https://github.com/Qiaoyi-Li) build the main architecture and the original version.
- [Junsen Wang](https://github.com/phyjswang) will maintain the algorithms for spin systems.

### How to contribute
- If you benefit from this package, please star it.
- If you find a bug, you can submit an issue or a PR if you have fixed it.
- Add a new instance to LocalSpace folder and submit a PR. Note you should add some comments and document it so that others can easily use it.
- If you are familiar with multi-processing parallelism in Julia and want to join us to improve the package, please contact us!  

## Acknowledgments
- This package uses [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) to implement basic tensor operations.
- This package uses [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) as Lanczos solver in DMRG and TDVP.
- I benefit a lot from reading the source code of [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl) and [ITensors.jl](https://github.com/ITensor/ITensors.jl).
