const GlobalTimer = TimerOutput()

# provide some unexported functions to control the parallel computing
get_num_threads_julia() = Threads.nthreads()
get_num_threads_mkl() = BLAS.get_num_threads()
set_num_threads_mkl(n::Int64) = BLAS.set_num_threads(n)
get_num_workers() = Distributed.nworkers()

