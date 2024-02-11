function __init__()

     println("FiniteMPS Version $(pkgversion(FiniteMPS))")

     # default multi-threading initialization
     _init_multithreading()

     return nothing
end

function _init_multithreading()


     # if MKL is not used, set num_threads of BLAS to 1 to avoid the confliction to the high-level multi-threading implementations
     if !any(x -> startswith(x, "libmkl_rt"),
          basename(lib.libname) for lib in BLAS.get_config().loaded_libs)
          BLAS.set_num_threads(1)
     end

     # close Strided in TensorKit
     TensorKit.Strided.set_num_threads(1)

     # initialize global variables
     global GlobalNumThreads_action = Threads.nthreads()

     # print
     println("Multi-threading Info:")
     println(" Julia: $(Threads.nthreads())")
     println(" BLAS: $(BLAS.get_num_threads())")
     println(" action: $(get_num_threads_action())")

     println("BLAS Info:")
     println(" $(BLAS.get_config())")
     return nothing
end