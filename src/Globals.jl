# provide some unexported functions to control the parallel computing
global GlobalThreadsExecutor = ThreadedEx(;simd = true, basesize = 1)
get_num_threads_julia() = Threads.nthreads()
get_num_threads_mkl() = BLAS.get_num_threads()
set_num_threads_mkl(n::Int64) = BLAS.set_num_threads(n)
get_num_workers() = Distributed.nworkers()

global GlobalThreads_action
function set_num_threads_action(n::Int64)
     @assert n ≥ 1
     n > Threads.nthreads() && @warn "n > Threads.nthreads() = $(Threads.nthreads()), not suggested!"
     global GlobalThreads_action = n
     return nothing
end
function get_num_threads_action()   
     # default = Threads.nthreads()
     !isdefined(FiniteMPS, :GlobalThreads_action) && set_num_threads_action(Threads.nthreads())
     return GlobalThreads_action
end





