function __init__()

     println("Julia Version $(VERSION)")
     println("FiniteMPS Version $(pkgversion(FiniteMPS))")

     # default multi-threading initialization
     _init_multithreading()

     # setup TensorKit caches
     TensorKit.DEFAULT_GLOBALCACHE_SIZE[] = 10^3
     for (_, cache) in TensorKit.GLOBAL_CACHES
          resize!(cache; maxsize=TensorKit.DEFAULT_GLOBALCACHE_SIZE[])
     end

     return nothing
end

function _init_multithreading()

     # if MKL is not used, close BLAS parallelism to avoid conflicting with outer Julia tasks
     _is_mkl_backend() || BLAS.set_num_threads(1)

     # close Strided in TensorKit
     TensorKit.Strided.disable_threads()

     # initialize global variables
     global GlobalNumThreads_action = Threads.nthreads(:default)

     # print
     println("Multi-threading Info:")
     println(" Julia: $(Threads.nthreads(:default))")
     println(" BLAS domain: $(BLAS.get_num_threads())")
     println(" action: $(get_num_threads_action())")
     println(" SVD MKL local: $(something(get_num_threads_svd_mkl(), "inherit"))")

     println("BLAS Info:")
     println(" $(BLAS.get_config())")
     return nothing
end
