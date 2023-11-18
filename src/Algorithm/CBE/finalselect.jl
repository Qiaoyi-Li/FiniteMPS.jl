function _finalselect(lsEl::SparseLeftTensor, RO::RightOrthComplement{N}) where {N}
     # contract truncated left environment tensor to the right one

     function f(El::LocalLeftTensor{2}, Er::LocalRightTensor{2})
          return El.A * Er.A
     end
     function f(El::LocalLeftTensor{3}, Er::LocalRightTensor{3})
          @tensor tmp[a; d] := El.A[a b c] * Er.A[c b d]
          return tmp
     end
     function f(El::LocalLeftTensor{2}, Ar::MPSTensor{4})
          @tensor tmp[a; c d e] := El.A[a b] * Ar.A[b c d e]
          return tmp
     end
     function f(El::LocalLeftTensor{3}, Ar::MPSTensor{5})
          @tensor tmp[a; d e f] := El.A[a b c] * Ar.A[c d e b f]
          return tmp
     end
     f(::Nothing, ::LocalRightTensor) = nothing
     f(::Nothing, ::MPSTensor) = nothing

     if get_num_workers() > 1 # multi-processing
     # TODO

     else # multi-threading
          Er, Ar = let f = f
               @floop GlobalThreadsExecutor for i in 1:N
                    Er_i = f(lsEl[i], RO.Er[i])
                    Ar_i = f(lsEl[i], RO.Ar[i])
                    @reduce() do (Er = nothing; Er_i), (Ar = nothing; Ar_i)
                         Er = axpy!(true, Er_i, Er)
                         Ar = axpy!(true, Ar_i, Ar)
                    end
               end
               Er, Ar
          end
     end

     # orthogonal projection, Ar - Er*Ar_c
     R = map(x -> rank(RO.Ar_c, x), 1:2)
     Ar_c = R[1] > 1 ? permute(RO.Ar_c.A, (1,), Tuple(2:R[1] + R[2])) : RO.Ar_c.A 

     return normalize!(axpy!(-1, Er * Ar_c, Ar))
end
