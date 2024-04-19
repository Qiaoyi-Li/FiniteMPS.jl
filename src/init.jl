function __init__()

     println("Julia Version $(VERSION)")
     # throw a warning for v1.10
     if VERSION ≥ v"1.10"
          @warn "It seems the new GC algorithm after Julia v1.10.0 will lead to bad performance of multi-threading parallelism in this package, v1.9 is strongly recommended!"
     end

     println("FiniteMPS Version $(pkgversion(FiniteMPS))")

     # default multi-threading initialization
     _init_multithreading()

     return nothing
end

function _init_multithreading()

     # check if julia is started with multiple threads
     if Threads.nthreads() == 1
          @warn "Julia is started with a single thread, some methods may not work!"
     end

     # if MKL is not used, set num_threads of BLAS to 1 to avoid the confliction to the high-level multi-threading implementations
     if !any(x -> startswith(x, "libmkl_rt"),
          basename(lib.libname) for lib in BLAS.get_config().loaded_libs)
          BLAS.set_num_threads(1)
     else
          # check if n_mkl * n_threads ≤ n_cpus
          if BLAS.get_num_threads() * Threads.nthreads() > get_num_cpus() 
          @warn "n_threads * n_mkl > n_cpus, may lead to bad performance!" 
          end
     end

     # close Strided in TensorKit
     TensorKit.Strided.set_num_threads(1)

     # initialize global variables
     global GlobalNumThreads_action = Threads.nthreads()
     global GlobalNumThreads_mul = 1
     global GlobalNumThreads_svd = Threads.nthreads()
     global GlobalNumThreads_eig = Threads.nthreads()


     # print
     println("Multi-threading Info:")
     println(" Julia: $(Threads.nthreads())")
     println(" BLAS: $(BLAS.get_num_threads())")
     println(" action: $(get_num_threads_action())")
     println(" mul: $(get_num_threads_mul())")
     println(" svd: $(get_num_threads_svd())")
     println(" eig: $(get_num_threads_eig())")

     println("BLAS Info:")
     println(" $(BLAS.get_config())")
     return nothing
end