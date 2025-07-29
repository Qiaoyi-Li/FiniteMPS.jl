# Multi-threading

In order to accelerate the computation via multi-threading parallel, please make sure Julia is started with [multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading). A warning will be thrown to remind you when loading `FiniteMPS` with a single thread.

If the linear algebra backend is `OpenBlas`, nested multi-threading is forbidden (a related discussion [here](https://carstenbauer.github.io/ThreadPinning.jl/stable/explanations/blas/)). So just close the parallelism of BLAS via
```julia
BLAS.set_num_threads(1)
```

When using MKL as the linear algebra backend, you can set the number of blas threads similarly, in which case nested multi-threading is allowed. Note MKL is invalid for some cpus and you should also close the BLAS parallelism in such case. You can follow [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl) to check if MKL is loaded successfully or not.

Please make sure the total threads number is not larger than the cpu cores (physical, without hyper-threading). Otherwise, the performance will become much worse due to conflict. The total threads can be estimated as    
```julia
Threads.nthreads() * BLAS.get_num_threads()
```   
