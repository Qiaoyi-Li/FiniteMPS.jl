using MKL, FiniteMPS

numthreads = 4
FiniteMPS.set_num_threads_mkl(1)
FiniteMPS.set_num_threads_action(numthreads)

# TODO

