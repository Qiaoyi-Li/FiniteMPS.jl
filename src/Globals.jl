# provide some unexported functions to control the parallel computing
get_num_cpus() = Sys.CPU_THREADS
get_num_threads_julia() = Threads.nthreads()
get_num_threads_mkl() = BLAS.get_num_threads()
set_num_threads_mkl(n::Int64) = BLAS.set_num_threads(n)
get_num_workers() = Distributed.nworkers()

global GlobalNumThreads_action::Int64
function set_num_threads_action(n::Int64)
     @assert n ≥ 1
     n > Threads.nthreads() && @warn "n > Threads.nthreads() = $(Threads.nthreads()), not suggested!"
     global GlobalNumThreads_action = n
     return nothing
end
get_num_threads_action() = GlobalNumThreads_action

global GlobalNumThreads_mul::Int64 = 1 # initialize it here since LocalSpace will use it before __init__
function set_num_threads_mul(n::Int64)
     @assert n ≥ 1
     n > Threads.nthreads() && @warn "n > Threads.nthreads() = $(Threads.nthreads()), not suggested!"
     global GlobalNumThreads_mul = n
     return nothing
end
get_num_threads_mul() = GlobalNumThreads_mul

global GlobalNumThreads_svd::Int64
function set_num_threads_svd(n::Int64)
     @assert n ≥ 1
     n > Threads.nthreads() && @warn "n > Threads.nthreads() = $(Threads.nthreads()), not suggested!"
     global GlobalNumThreads_svd = n
     return nothing
end
get_num_threads_svd() = GlobalNumThreads_svd

global GlobalNumThreads_eig::Int64
function set_num_threads_eig(n::Int64)
     @assert n ≥ 1
     n > Threads.nthreads() && @warn "n > Threads.nthreads() = $(Threads.nthreads()), not suggested!"
     global GlobalNumThreads_eig = n
     return nothing
end
get_num_threads_eig() = GlobalNumThreads_eig





