module FiniteMPSMKLExt

using FiniteMPS
using MKL_jll: libmkl_rt

const SVD_GATE = Base.Semaphore(1)

function FiniteMPS._with_svd_threads(f, n::Int)
    FiniteMPS._is_mkl_backend() || return f()
    return Base.acquire(SVD_GATE) do
        previous = ccall((:mkl_set_num_threads_local, libmkl_rt), Cint, (Cint,), n)
        try
            return f()
        finally
            ccall((:mkl_set_num_threads_local, libmkl_rt), Cint, (Cint,), previous)
        end
    end
end

end
