# provide some unexported functions to control the parallel computing
get_num_threads_julia() = Threads.nthreads()
get_num_workers() = Distributed.nworkers()

global GlobalNumThreads_action::Int = 1
function set_num_threads_action(n::Int)
     @assert n ≥ 1
     n > Threads.nthreads() && @warn "n > Threads.nthreads() = $(Threads.nthreads()), not suggested!"
     global GlobalNumThreads_action = n
     return nothing
end
get_num_threads_action() = GlobalNumThreads_action

global GlobalNumThreads_svd_mkl::Union{Nothing,Int} = nothing
function set_num_threads_svd_mkl(n::Union{Nothing,Int})
     isnothing(n) || n > 0 || throw(ArgumentError("SVD MKL thread count must be positive or nothing"))
     global GlobalNumThreads_svd_mkl = n
     return nothing
end
get_num_threads_svd_mkl() = GlobalNumThreads_svd_mkl

_is_mkl_backend() = any(lib -> startswith(basename(lib.libname), "libmkl_rt"), BLAS.get_config().loaded_libs)
_with_svd_threads(f) = _with_svd_threads(f, GlobalNumThreads_svd_mkl)
_with_svd_threads(f, ::Union{Nothing,Integer}) = f()

_isabsent(x) = x === nothing
_accumulate_owned(acc, term) = _isabsent(acc) ? term : add!!(acc, term)
