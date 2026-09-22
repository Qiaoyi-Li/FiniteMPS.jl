# Multi-threading

Start Julia with [multiple threads](https://docs.julialang.org/en/v1/manual/multi-threading/#man-multithreading) for the outer FiniteMPS tasks. For example, `julia -t8,0 --gcthreads=4,0` uses eight default worker threads and four GC mark threads. FiniteMPS requires Julia 1.11.5 or later.

FiniteMPS disables Strided threading and sets a non-MKL BLAS backend to one thread to avoid competing with the outer tasks. The ordinary BLAS-domain setting is controlled explicitly with
```julia
BLAS.set_num_threads(1)
```

When using [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl), the process-wide MKL baseline remains under the job's configuration, such as `MKL_NUM_THREADS`. FiniteMPS does not change that baseline during an SVD. An optional setting applies only to the current synchronous SVD call and restores the previous native thread-local value afterwards:
```julia
FiniteMPS.set_num_threads_svd_mkl(4)
FiniteMPS.set_num_threads_svd_mkl(nothing) # inherit the current global setting
```

The SVD setting takes effect only with an active MKL backend and the MKL_jll extension. Enhanced SVD calls share a concurrency gate; other contractions continue to use the ordinary backend settings. The total core budget belongs to the job configuration: multiplying Julia threads by the BLAS-domain thread count does not account for LAPACK, thread-local settings or the scheduler's allocated cores.

Keep TensorKit's internal debug timers disabled during parallel FiniteMPS execution. Enable them only for a serial diagnostic workload, since TensorKit shares one global timer across calls. FiniteMPS's own task-local timers are merged after the corresponding workers complete.
