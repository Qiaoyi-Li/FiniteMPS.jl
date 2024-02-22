function _preselect(LO::LeftOrthComplement{N}) where {N}

     function f(Al::MPSTensor{3}, iso::AbstractTensorMap)
          @tensor tmp[a b; f] := Al.A[a b e] * iso[e f]
          return tmp
     end
     function f(Al::MPSTensor{4}, iso::AbstractTensorMap)
          if rank(iso) == 2
               @tensor tmp[a b; c f] := Al.A[a b c e] * iso[e f]
          else
               # MPS tensor with an additional bond
               @tensor tmp[a b; f] := Al.A[a b d e] * iso[d e f]
          end
          return tmp
     end
     function f(Al::MPSTensor{5}, iso::AbstractTensorMap)
          @tensor tmp[a b; c f] := Al.A[a b c d e] * iso[d e f]
          return tmp
     end
     f(::Nothing, ::AbstractTensorMap) = nothing

     # fuse the horizontal bonds of MPS and H
     lsIso = fuse(LO.El)

     if get_num_workers() > 1 # multi-processing
          Al_oc = @distributed (add!) for i in 1:N
               f(LO.Al[i] - MPSTensor(_leftProj(LO.El[i], LO.Al_c)), lsIso[i])
          end
     else

          Al_oc = nothing
          Lock = Threads.ReentrantLock()
          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true
                    i = Threads.atomic_add!(idx, 1)
                    i > N && break

                    axpy!(-1, _leftProj(LO.El[i], LO.Al_c), LO.Al[i].A) # project to orthogonal complement
                    Al_i = f(LO.Al[i], lsIso[i])

                    lock(Lock)
                    try
                         Al_oc = axpy!(true, Al_i, Al_oc)
                    catch
                         rethrow()
                    finally
                         unlock(Lock)
                    end
               end
          end
     end

     return normalize!(Al_oc)
end

function _preselect(RO::RightOrthComplement{N}) where {N}

     function f(Ar::MPSTensor{3}, iso::AbstractTensorMap)
          @tensor tmp[a d; f] := iso[a b] * Ar.A[b d f]
          return tmp
     end
     function f(Ar::MPSTensor{4}, iso::AbstractTensorMap)
          if rank(iso) == 2
               @tensor tmp[a d; e f] := iso[a b] * Ar.A[b d e f]
          else
               # MPS tensor with an additional bond
               @tensor tmp[a d; f] := iso[a b c] * Ar.A[b d c f]
          end
          return tmp
     end
     function f(Ar::MPSTensor{5}, iso::AbstractTensorMap)
          @tensor tmp[a d; e f] := iso[a b c] * Ar.A[b d e c f]
          return tmp
     end
     f(::Nothing, ::AbstractTensorMap) = nothing

     # fuse the horizontal bonds of MPS and H
     lsIso = fuse(RO.Er)

     if get_num_workers() > 1 # multi-processing
          Ar_oc = @distributed (add!) for i in 1:N
               f(RO.Ar[i] - MPSTensor(_rightProj(RO.Er[i], RO.Ar_c)), lsIso[i])
          end
         
     else

          Ar_oc = nothing
          Lock = Threads.ReentrantLock()
          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true
                    i = Threads.atomic_add!(idx, 1)
                    i > N && break

                    axpy!(-1, _rightProj(RO.Er[i], RO.Ar_c), RO.Ar[i].A) # project to orthogonal complement
                    Ar_i = f(RO.Ar[i], lsIso[i])

                    lock(Lock)
                    try
                         Ar_oc = axpy!(true, Ar_i, Ar_oc)
                    catch
                         rethrow()
                    finally
                         unlock(Lock)
                    end
               end
          end
     end

     return  normalize!(Ar_oc)
end