# FiniteMPS.jl

This repository provides some MPS-based algorithms such as DMRG and TDVP(TODO).

Note it is still in the initial stage, only supports multi-threading. We may add more algorithms and try to support multi-processing in the future.

`Warning:` There may exist undiscovered bugs, therefore, please use it at your own risk. Benchmark with exact diagonalization (ED) is recommended before using it to simulate any new models.   

## Installation
To install this package, you can press "]" in REPL and then type
```julia
pkg> add "https://github.com/Qiaoyi-Li/FiniteMPS.jl.git"
```

## Demo
We provide a demonstration of DMRG computation of Hubbard model.
1. cd to "example" folder and then open julia REPL.
2. activate and instantiate. Note doing this will add [FiniteLattices.jl](https://github.com/Qiaoyi-Li/FiniteLattices.jl), our another package used to generate nearest neighbor pairs on finite square lattices.
```julia
pkg> activate .
pkg> instantiate
```
3. run script "HubbardDMRG_threads.jl".
```julia
include("HubbardDMRG_threads.jl")
```
`Warning:` Note we do not consider single-threading case in this version. Please make sure julia is started with [multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading). Otherwise, some errors may occur.

## Features
We use a tree-like structure, namely `InteractionTree` in this package, to help generating a sparse Hamiltonian MPO via automata and calculating observables.

## Acknowledgments

- This package uses [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) to implement basic tensor operations.

- The architecture mainly borrows from [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl), which provides much more functionalities for infinite MPS. 
- Package [SerializedElementArrays.jl](https://github.com/ITensor/SerializedElementArrays.jl) is used to support storing local tensors in disk.