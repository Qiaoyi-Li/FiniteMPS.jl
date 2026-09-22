using MKL
MKL.set_num_threads(2)
using FiniteMPS

_mkl_local(n) = ccall((:mkl_set_num_threads_local, MKL.libmkl_rt), Cint, (Cint,), n)
_mkl_threads() = ccall((:mkl_get_max_threads, MKL.libmkl_rt), Cint, ())

@testset "MKL local SVD threads" begin
    initialized = (BLAS.get_num_threads(), _mkl_threads())
    MKL.set_num_threads(1)
    previous = _mkl_local(2)
    FiniteMPS.set_num_threads_svd_mkl(4)
    try
        entered = FiniteMPS._with_svd_threads(_mkl_threads)
        restored = _mkl_local(0)
        @test (initialized, entered, restored, _mkl_threads()) == ((2, 2), 4, 2, 1)

        _mkl_local(2)
        entered_on_error = Ref{Cint}(0)
        @test_throws ErrorException FiniteMPS._with_svd_threads() do
            entered_on_error[] = _mkl_threads()
            error("SVD scope failed")
        end
        restored = _mkl_local(0)
        @test (entered_on_error[], restored, _mkl_threads()) == (4, 2, 1)
    finally
        FiniteMPS.set_num_threads_svd_mkl(nothing)
        _mkl_local(previous)
    end
end
