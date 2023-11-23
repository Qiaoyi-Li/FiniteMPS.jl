# FiniteMPS.jl

This package provides some MPS-based algorithms. Compared with the well-known [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl), we focus on finite MPS (i.e., with finite length and open boundaries) and the performance. Now it contains
- Density matrix normalization group (DMRG) to search ground states. 
- Time-dependent variational principle ([TDVP](https://doi.org/10.1103/PhysRevB.94.165116)) for time evolution.
- Series-expansion thermal tensor network ([SETTN](https://doi.org/10.1103/PhysRevB.95.161104)) and tangent space tensor renormalization group ([tanTRG](https://doi.org/10.1103/PhysRevLett.130.226502)) for purification-based finite-temperature simulations.
- Non-abelian symmetries can be equiped thanks to [TensorKit.jl](https://github.com/Jutho/TensorKit.jl).
- Controlled bond expansion ([CBE](https://doi.org/10.1103/PhysRevLett.130.246402)) technique to improve the 1-TDVP. Note some key methods for MPS version are not yet implemented, to be added in the future.


`Warning:` There may exist undiscovered bugs, therefore, please use it at your own risk. Benchmark with exact diagonalization (ED) is recommended before using it to simulate any new models.   

## Installation
To install this package, you can press "]" in REPL and then type
```julia
pkg> add "https://github.com/Qiaoyi-Li/FiniteMPS.jl.git"
```

## Demo
### DMRG computation of Hubbard model
1. cd to "example" folder and then open julia REPL.
2. activate and instantiate. Note doing this will add [FiniteLattices.jl](https://github.com/Qiaoyi-Li/FiniteLattices.jl), our another package used to generate nearest neighbor pairs on finite square lattices.
```julia
pkg> activate .
pkg> instantiate
```
`Note:` Some methods in TensorKit.jl are replaced in order to be  compatible with parallel computing, thus it will throw some warnings when compiling. Just ignore them. 

3. run script "HubbardDMRG_threads.jl".
```julia
include("HubbardDMRG_threads.jl")
```

### tanTRG approach to thermodynamics of free fermions
Prepare environment similarly and run the script 
```julia
include("FreeFermion_tanTRG.jl")
```
`Warning:` Note we do not consider single-threading case in this version. Please make sure julia is started with [multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading). Otherwise, some errors may occur.

## Acknowledgments
- This package uses [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) to implement basic tensor operations.
- The architecture mainly borrows from [MPSKit.jl](https://github.com/maartenvd/MPSKit.jl), which provides much more functionalities for infinite MPS. 
- Package [SerializedElementArrays.jl](https://github.com/ITensor/SerializedElementArrays.jl) is used to support storing local tensors in disk.