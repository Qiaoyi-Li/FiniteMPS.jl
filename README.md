# FiniteMPS.jl

This repository provide some MPS-based algorithms such as DMRG and TDVP(TODO).

Note it is still in the initial stage, only supports multi-threading. We may add more algorithms and try to support multi-processing in the future.  

## Installation
To install this package, you can press "]" in REPL and then type
```julia
pkg> add "https://github.com/Qiaoyi-Li/FiniteMPS.jl.git"
```

## Features
We use a tree-like structure, namely `InteractionTree` in this package, to help generating a sparse Hamiltonian MPO via automata and calculating observables.

## Acknowledgments

- This package uses [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) to implement basic tensor operations.

- The architecture mainly borrows from [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl), which provides much more functionalities for infinite MPS. 
- Package [SerializedElementArrays.jl](https://github.com/ITensor/SerializedElementArrays.jl) is used to support storing local tensors in disk.