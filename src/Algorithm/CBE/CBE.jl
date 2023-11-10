""" 
     abstract type CBEAlgorithm

Abstract type of all (controlled bond expansion) CBE algorithm.
"""
abstract type CBEAlgorithm end

"""
     struct StandardCBE <: CBEAlgorithm 
          direction::Symbol
          D::Int64
          tol::Int64
     end

Standard CBE algorithm, details see [PhysRevLett.130.246402](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.246402).
"""
struct StandardCBE <: CBEAlgorithm 
     direction::Symbol
     D::Int64
     tol::Float64
end


function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor, Alg::StandardCBE; kwargs...)
     
     if Alg.direction == :L # CBE for left to right sweep
          # bond-canonicalize Al
          Al, S = leftorth(Al; trunc = truncbelow(Alg.tol))
      


     else # CBE for right to left sweep
          # bond-canonicalize Ar
          S, Ar_rc::MPSTensor = rightorth(Ar; trunc = truncbelow(Alg.tol))
          # orthogonal complement
          Al_oc = _CBE_leftnull(Al)
          Ar_oc = _CBE_rightnull(Ar)
          
          Er = _pushleft(PH.Er, Ar_oc', PH.H[2], Ar)
          # 1-st svd, implemented by eig
          C::MPSTensor, info1 = _CBE_MMd2US(Er; trunc = truncbelow(Alg.tol), normalize = true)
          Al_us::MPSTensor = Al*C

          El = _pushright(PH.El, Al_oc', PH.H[1], Al_us)
          # 2-nd svd, D -> D/w
          El_trunc, info2 = _CBE_MdM2US(El; trunc = truncbelow(Alg.tol) & truncdim(div(Alg.D, length(El))), normalize = true)

          # 3-rd svd
          U::MPSTensor, info3 = _CBE_MMd2U(El_trunc; trunc = truncbelow(Alg.tol) & truncdim(Alg.D), normalize = true)
          Al_pre::MPSTensor = Al_oc * U

          # final select
          El_final = _pushright(PH.El, Al_pre', PH.H[1], Al)
          M::MPSTensor = _final_contract(El_final, Er)

          # 4-th svd, directly use svd 
          D₀ = dim(Ar, 1)[2] # original bond dimension
          Q::MPSTensor, ~, info4 = leftorth(M; trunc = truncbelow(Alg.tol) & truncdim(Alg.D - D₀))
          Al_final::MPSTensor = Al_pre * Q

          # TODO return updated Al, Ar and info
     end

end

_CBE_leftnull(Al::MPSTensor{3})::MPSTensor{3} = leftnull(Al, (1, 2), (3,))
_CBE_leftnull(Al::MPSTensor{4})::MPSTensor{4} = leftnull(Al, (1, 2, 3), (4,))
_CBE_rightnull(Ar::MPSTensor{3})::MPSTensor{3} = rightnull(Ar, (1,), (2, 3)) 
_CBE_rightnull(Ar::MPSTensor{4})::MPSTensor{4} = rightnull(Ar, (1,), (2, 3, 4))

function _final_contract(lsEl::SparseLeftTensor, lsEr::SparseRightTensor)
     # final contract, b is summed over manually
     #    --c    c--
     #   |          |
     #  El--b    b--Er
     #   |          |
     #    --a    d-- 
     function _final_contract_single(El::LocalLeftTensor{2}, Er::LocalRightTensor{2})
          return El.A * Er.A
     end
     function _final_contract_single(El::LocalLeftTensor{3}, Er::LocalRightTensor{3})
          @tensor tmp[a; d] := El.A[a b c] * Er.A[c b d]
     end
     _final_contract_single(::Nothing, ::LocalRightTensor) = nothing
     _final_contract_single(::LocalLeftTensor, ::Nothing) = nothing

     if get_num_workers() > 1 # multi-processing
     # TODO

     else # multi-threading
          @floop GlobalThreadsExecutor for (El, Er) in zip(lsEl, lsEr)
               tmp = _final_contract_single(El, Er)
               @reduce() do (M = nothing; tmp)
                    M = axpy!(true, tmp, M)
               end
          end
     end
     
     return M
end



