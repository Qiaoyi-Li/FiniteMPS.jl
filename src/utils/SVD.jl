function _raw_svd(t::AbstractTensorMap; kwargs...)
     return _raw_svd_copy(() -> copy(t); kwargs...)
end
function _raw_svd(t::AbstractTensorMap, p::Index2Tuple; kwargs...)
     return _raw_svd_copy(() -> permute(t, p; copy = true); kwargs...)
end

function _raw_svd_copy(makecopy;
     trunc = notrunc(), alg = MatrixAlgebraKit.DivideAndConquer(fixgauge = false))
     t = makecopy()
     try
          return _raw_svd!(t, trunc, alg)
     catch err
          (alg isa MatrixAlgebraKit.DivideAndConquer &&
               err isa LinearAlgebra.LAPACKException && err.info > 0) || rethrow()
          @warn "Divide-and-conquer SVD did not converge; retrying with QR iteration."
          t = makecopy()
          return _raw_svd!(t, trunc, MatrixAlgebraKit.QRIteration(fixgauge = false))
     end
end

function _raw_svd!(t, trunc, alg)
     if trunc == notrunc()
          U, S, Vᴴ = _with_svd_threads() do
               TensorKit.svd_compact!(t; alg)
          end
          return U, S, Vᴴ, zero(scalartype(S))
     else
          return _with_svd_threads() do
               TensorKit.svd_trunc!(t; trunc, alg)
          end
     end
end
