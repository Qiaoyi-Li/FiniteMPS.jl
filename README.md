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
- We provide a hamiltonian generator via automata (like `OpSum` in [ITensors.jl](https://github.com/ITensor/ITensors.jl)) so that the interactions can be added in a simple and general way, details please see the workflow section.
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

## Suggested workflow

### Installation
To install this package, you can press "]" in REPL and type
```julia
pkg> add "https://github.com/Qiaoyi-Li/FiniteMPS.jl.git"
```

### Multi-threading settings
- Only multi-threading is stable in current version, please make sure Julia is started with [multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading). If this is not satisfied, it will throw an warning to remind you when loading `FiniteMPS`. 
  
  TODO: I will add single-thread mode and try to support multi-processing in the future.

- If the linear algebra backend is `OpenBlas`, nested multi-threading is forbidden (a related discussion [here](https://carstenbauer.github.io/ThreadPinning.jl/stable/explanations/blas/)). So just close the parallelism of BLAS via
  ```julia
     BLAS.set_num_threads(1)
  ```

- If using MKL as the linear algebra backend, you can set the number of blas threads similarly and nested multi-threading can be supported in this case. Note MKL is invalid for some cpus and you should also close the BLAS parallelism in this case. You can follow [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl)) to check if MKL is loaded successfully or not.
  
   `Warning`: Please make sure the total threads number 
   ```julia
       Threads.nthreads() * BLAS.get_num_threads()
   ```   
   is not larger than the cpu cores (physical, without hyper-threading). Otherwise, the performance will become much worse due to confliction.


### Local space
Choose a local space, i.e. the local 1-site Hilbert space and some 1-site operators according to the model and symmetry used. We predefine some in folder `LocalSpace` and you can use the documentation to see which operators are provided, e.g 
```julia 
help?> U₁Spin
```
Note: It is impossible to predefine all local spaces so you must write a new one if that you need is not in it. Moreover, some `no method` errors may occur for new defined local spaces due to the multiple-dispatched implementations may not cover all cases. Please feel free to submit an issue if this happens and I will deal with it as soon as possible.

### Hamiltonian
After that, you need to tell the program the interactions in hamiltonian. We use a tree-like struct `InteractionTree` to store them and provide an interface named `addIntr!` to add terms. The following is a demo code to generate the hamiltonian of an AFM Heisenberg chain.
```julia
     L = 8
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
               name = (:Sz, :Sz))
          addIntr!(Root,
               U₁Spin.S₋₊,
               (i, i+1),
               0.5;
               name = (:Sz, :Sz))
     end
     Ham = AutomataMPO(InteractionTree(Root))
```

### Main algorithm
TODO

### Observables
TODO

`Warning:` There may exist undiscovered bugs, therefore, please use it at your own risk. Benchmark with exact diagonalization (ED) is recommended before using it to simulate any new models.

## Contributions
TODO

## Acknowledgments
- This package uses [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) to implement basic tensor operations.
- This package uses [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) as Lanczos solver in DMRG and TDVP.
- I benefit a lot from reading the source code of [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl) and [ITensors.jl](https://github.com/ITensor/ITensors.jl).
