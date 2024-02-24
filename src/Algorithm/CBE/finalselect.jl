function _finalselect(lsEl::SparseLeftTensor, RO::RightOrthComplement{N}) where {N}
     # contract truncated left environment tensor to the right one

     function f(El::LocalLeftTensor{2}, Er::LocalRightTensor{2})
          return El.A * Er.A
     end
     function f(El::LocalLeftTensor{3}, Er::LocalRightTensor{3})
          @tensor tmp[a; d] := El.A[a b c] * Er.A[c b d]
          return tmp
     end
     function f(El::LocalLeftTensor{2}, Ar::MPSTensor{3})
          @tensor tmp[a c; e] := El.A[a b] * Ar.A[b c e]
          return tmp
     end
     function f(El::LocalLeftTensor{2}, Ar::MPSTensor{4})
          @tensor tmp[a c; d e] := El.A[a b] * Ar.A[b c d e]
          return tmp
     end
     function f(El::LocalLeftTensor{3}, Ar::MPSTensor{4})
          @tensor tmp[a d; f] := El.A[a b c] * Ar.A[c d b f]
          return tmp
     end
     function f(El::LocalLeftTensor{3}, Ar::MPSTensor{5})
          @tensor tmp[a d; e f] := El.A[a b c] * Ar.A[c d e b f]
          return tmp
     end
     f(::Nothing, ::LocalRightTensor) = nothing
     f(::Nothing, ::MPSTensor) = nothing

     if get_num_workers() > 1 # multi-processing
          _add = (x, y) -> (add!(x[1], y[1]), add!(x[2], y[2]))
          Er, Ar = let f = f
               @distributed (_add) for i in 1:N
                    f(lsEl[i], RO.Er[i]), f(lsEl[i], RO.Ar[i])
               end
          end

     else # multi-threading
          
          Er = nothing
          Ar = nothing
          Lock = Threads.ReentrantLock()
          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true
                    idx_t = Threads.atomic_add!(idx, 1)
                    idx_t > N && break

                    Er_i = f(lsEl[idx_t], RO.Er[idx_t])
                    Ar_i = f(lsEl[idx_t], RO.Ar[idx_t])

                    lock(Lock)
                    try
                         Er = axpy!(true, Er_i, Er)
                         Ar = axpy!(true, Ar_i, Ar)
                    catch
                         rethrow()
                    finally
                         unlock(Lock)
                    end
               end
          end  

     end

     # orthogonal projection, Ar - Er*Ar_c
     return normalize!(axpy!(-1, _rightProj(LocalRightTensor(Er), RO.Ar_c), Ar))
end

function _finalselect(LO::LeftOrthComplement{N}, lsEr::SparseRightTensor) where {N}
     # contract truncated right environment tensor to the left one

     function f(El::LocalLeftTensor{2}, Er::LocalRightTensor{2})
          return El.A * Er.A
     end
     function f(El::LocalLeftTensor{3}, Er::LocalRightTensor{3})
          @tensor tmp[a; d] := El.A[a b c] * Er.A[c b d]
          return tmp
     end
     function f(Al::MPSTensor{3}, Er::LocalRightTensor{2})
          @tensor tmp[a b; f] := Al.A[a b e] * Er.A[e f]
          return tmp
     end
     function f(Al::MPSTensor{4}, Er::LocalRightTensor{2})
          @tensor tmp[a b; c f] := Al.A[a b c e] * Er.A[e f]
          return tmp
     end
     function f(Al::MPSTensor{4}, Er::LocalRightTensor{3})
          @tensor tmp[a b; f] := Al.A[a b d e] * Er.A[e d f]
          return tmp
     end
     function f(Al::MPSTensor{5}, Er::LocalRightTensor{3})
          @tensor tmp[a b; c f] := Al.A[a b c d e] * Er.A[e d f]
          return tmp
     end
     f(::LocalRightTensor, ::Nothing) = nothing
     f(::MPSTensor, ::Nothing) = nothing

     if get_num_workers() > 1 # multi-processing
          _add = (x, y) -> (add!(x[1], y[1]), add!(x[2], y[2]))
          El, Al = let f = f
               @distributed (_add) for i in 1:N
                    f(LO.El[i], lsEr[i]), f(LO.Al[i], lsEr[i])
               end
          end
     else # multi-threading

          El = nothing
          Al = nothing
          Lock = Threads.ReentrantLock()
          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true
                    idx_t = Threads.atomic_add!(idx, 1)
                    idx_t > N && break

                    El_i = f(LO.El[idx_t], lsEr[idx_t])
                    Al_i = f(LO.Al[idx_t], lsEr[idx_t])

                    lock(Lock)
                    try
                         El = axpy!(true, El_i, El)
                         Al = axpy!(true, Al_i, Al)
                    catch
                         rethrow()
                    finally
                         unlock(Lock)
                    end
               end
          end

     end

     # orthogonal projection, Al - El*Al_c
     return normalize!(axpy!(-1, _leftProj(LocalLeftTensor(El), LO.Al_c), Al))
end