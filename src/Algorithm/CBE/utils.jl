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

function _directsum_Al(Al::MPSTensor{R}, Al_f::MPSTensor{R}) where {R}
     # Al ⊕ Al_f

     EmbA, EmbB = oplusEmbed(Al.A, Al_f.A, R)
     return permute(Al.A, Tuple(1:R-1), (R,)) * EmbA + permute(Al_f.A, Tuple(1:R-1), (R,)) * EmbB
end

function _directsum_Ar(Ar::MPSTensor{R}, Ar_f::MPSTensor{R}) where {R}
     # Ar ⊕ Ar_f

     EmbA, EmbB = oplusEmbed(Ar.A, Ar_f.A, 1)
     return EmbA * permute(Ar.A, (1,), Tuple(2:R)) + EmbB * permute(Ar_f.A, (1,), Tuple(2:R))
end

function _expand_Ar(Al::MPSTensor{3}, Al_ex::MPSTensor{3}, Ar::MPSTensor{3})::MPSTensor{3}
     @tensor Ar_ex[d e; f] := (Al.A[a b c] * Al_ex.A'[d a b]) * Ar.A[c e f]
     return Ar_ex
end
function _expand_Ar(Al::MPSTensor{4}, Al_ex::MPSTensor{4}, Ar::MPSTensor{4})::MPSTensor{4}
     @tensor Ar_ex[e f; g h] := (Al.A[a b c d] * Al_ex.A'[c e a b]) * Ar.A[d f g h]
     return Ar_ex
end

function _expand_Al(Ar::MPSTensor{3}, Ar_ex::MPSTensor{3}, Al::MPSTensor{3})::MPSTensor{3}
     @tensor Al_ex[a b; h] := (Ar_ex.A'[g h e] * Ar.A[d e f]) * Al.A[a b d]
     return Al_ex
end    

function _expand_Al(Ar::MPSTensor{4}, Ar_ex::MPSTensor{4}, Al::MPSTensor{4})::MPSTensor{4}
     @tensor Al_ex[a b; c h] := (Ar_ex.A'[f g h e] * Ar.A[d e f g]) * Al.A[a b c d]
     return Al_ex
end    

function _isometry_Al(Al::MPSTensor{3})::MPSTensor{3}
     fullspace = fuse(codomain(Al)[1], codomain(Al)[2])
     return isometry(codomain(Al)[1] ⊗ codomain(Al)[2], fullspace)
end

function _isometry_Al(Al::MPSTensor{4})::MPSTensor{4}
     fullspace = fuse(codomain(Al)[1] ⊗ codomain(Al)[2], domain(Al)[1]')
     return isometry(codomain(Al)[1] ⊗ codomain(Al)[2] ⊗ domain(Al)[1]', fullspace)
end

function _isometry_Ar(Ar::MPSTensor{3})::MPSTensor{3}
     fullspace = fuse(codomain(Ar)[2]', domain(Ar)[1])
     return isometry(fullspace, codomain(Ar)[2]' ⊗ domain(Ar)[1])
end

function _isometry_Ar(Ar::MPSTensor{4})::MPSTensor{4}
     fullspace = fuse(codomain(Ar)[2]', domain(Ar)[1] ⊗ domain(Ar)[2])
     return isometry(fullspace, codomain(Ar)[2]' ⊗ domain(Ar)[1] ⊗ domain(Ar)[2])
end