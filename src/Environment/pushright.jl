"""
     pushright!(::AbstractEnvironment)

Push right the given environment object, i.e. `Center == [i, j]` to `[i + 1, j]`.
"""
function pushright!(obj::SimpleEnvironment{L,2,T}) where {L,T<:Tuple{AdjointMPS,DenseMPS}}
     si = obj.Center[1]
     @assert si < L

     obj.El[si+1] = _pushright(obj.El[si], obj[1][si], obj[2][si])
     obj.Center[1] += 1
     return obj
end

function pushright!(obj::SparseEnvironment{L,3,T}) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     si = obj.Center[1]
     @assert si < L

     obj.El[si+1] = _pushright(obj.El[si], obj[1][si], obj[2][si], obj[3][si])
     obj.Center[1] += 1
     return obj
end

function _pushright(El::SparseLeftTensor, A::AdjointMPSTensor, H::SparseMPOTensor, B::MPSTensor; kwargs...)
     sz = size(H)
     El_next = SparseLeftTensor(nothing, sz[2])

     if get_num_workers() > 1 # multi-processing

          # use pmap to dispatch interactions
          valid_idx = [(i, j) for j in 1:sz[2] for i in filter(x -> !_isabsent(H[x, j]) && !_isabsent(El[x]), 1:sz[1])]
          lsEl = pmap(valid_idx) do (i, j)
               _pushright(El[i], A, H[i, j], B; sparse=true), j
          end

          for (El, j) in lsEl
               El_next[j] = _accumulate_owned(El_next[j], El)
          end

     else # multi-threading

          validIdx = [(i, j) for j in 1:sz[2] for i in filter(x -> !_isabsent(H[x, j]) && !_isabsent(El[x]), 1:sz[1])]

          Lock = Threads.ReentrantLock()
          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true
                    idx_t = Threads.atomic_add!(idx, 1)
                    idx_t > length(validIdx) && break

                    (i, j) = validIdx[idx_t]
                    El_i = _pushright(El[i], A, H[i, j], B; sparse=true)

                    lock(Lock)
                    try
                         El_next[j] = _accumulate_owned(El_next[j], El_i)
                    catch
                         rethrow()
                    finally
                         unlock(Lock)
                    end
               end
          end

     end

     return El_next

end

function _pushright(El::LocalLeftTensor{2}, A::AdjointMPSTensor{3}, B::MPSTensor{3}; kwargs...)
     if numout(A) == 1
          @tensor allocator = ManualAllocator() tmp[d; e] := (El.A[a b] * A.A[d a c]) * B.A[b c e]
     else
          @tensor allocator = ManualAllocator() tmp[d; e] := (El.A[a b] * A.A[c d a]) * B.A[b c e]
     end
     return LocalLeftTensor(tmp, El.tag)
end

function _pushright(El::LocalLeftTensor{2}, A::AdjointMPSTensor{3}, H::IdentityOperator, B::MPSTensor{3}; kwargs...)
     return _pushright(El, A, B) * H.strength[]
end

function _pushright(El::LocalLeftTensor{2}, A::AdjointMPSTensor{3}, H::LocalOperator{1,1}, B::MPSTensor{3}; kwargs...)
     if numout(A) == 1
          @tensor allocator = ManualAllocator() tmp[a; f] := ((A.A[a b c] * H.A[c e]) * El.A[b d]) * B.A[d e f]
     else
          @tensor allocator = ManualAllocator() tmp[a; f] := ((A.A[c a b] * H.A[c e]) * El.A[b d]) * B.A[d e f]
     end
     return LocalLeftTensor(tmp * H.strength[], El.tag)
end

function _pushright(El::LocalLeftTensor{2}, A::AdjointMPSTensor{3}, H::LocalOperator{1,2}, B::MPSTensor{3}; kwargs...)
     # χ < d in most sparse cases
     if get(kwargs, :sparse, true)
          # D^3d + D^2d^2χ + D^3dχ
          if numout(A) == 1
               @tensor allocator = ManualAllocator() tmp[a; f g] := ((A.A[a b c] * El.A[b d]) * H.A[c e f]) * B.A[d e g]
          else
               @tensor allocator = ManualAllocator() tmp[a; f g] := ((A.A[c a b] * El.A[b d]) * H.A[c e f]) * B.A[d e g]
          end
     else
          # D^3d + D^3d^2 + D^2d^2χ
          if numout(A) == 1
               @tensor allocator = ManualAllocator() tmp[a; f g] := ((A.A[a b c] * El.A[b d]) * B.A[d e g]) * H.A[c e f]
          else
               @tensor allocator = ManualAllocator() tmp[a; f g] := ((A.A[c a b] * El.A[b d]) * B.A[d e g]) * H.A[c e f]
          end
     end
     return LocalLeftTensor(tmp * H.strength[], (El.tag[1], H.tag[2][2], El.tag[2]))
end

function _pushright(El::LocalLeftTensor{3}, A::AdjointMPSTensor{3}, H::LocalOperator{2,1}, B::MPSTensor{3}; kwargs...)
     # contraction order of El and H does not affect complexity
     # D^3dχ + D^2d^2χ + D^3d
     if numout(A) == 1
          @tensor allocator = ManualAllocator() tmp[a; g] := ((A.A[a b c] * El.A[b d e]) * H.A[d c f]) * B.A[e f g]
     else
          @tensor allocator = ManualAllocator() tmp[a; g] := ((A.A[c a b] * El.A[b d e]) * H.A[d c f]) * B.A[e f g]
     end
     return LocalLeftTensor(tmp * H.strength[], (El.tag[1], El.tag[3]))
end

function _pushright(El::LocalLeftTensor{3}, A::AdjointMPSTensor{3}, H::LocalOperator{1,1}, B::MPSTensor{3}; kwargs...)
     # D^2d^2 + D^3dχ + D^3dχ
     if numout(A) == 1
          @tensor allocator = ManualAllocator() tmp[a; d g] := ((A.A[a b c] * H.A[c f]) * El.A[b d e]) * B.A[e f g]
     else
          @tensor allocator = ManualAllocator() tmp[a; d g] := ((A.A[c a b] * H.A[c f]) * El.A[b d e]) * B.A[e f g]
     end
     return LocalLeftTensor(tmp * H.strength[], El.tag)
end

function _pushright(El::LocalLeftTensor{3}, A::AdjointMPSTensor{3}, H::IdentityOperator, B::MPSTensor{3}; kwargs...)
     # D^2d^2 + D^3dχ + D^3dχ
     if numout(A) == 1
          @tensor allocator = ManualAllocator() tmp[a; d g] := (A.A[a b f] * El.A[b d e]) * B.A[e f g]
     else
          @tensor allocator = ManualAllocator() tmp[a; d g] := (A.A[f a b] * El.A[b d e]) * B.A[e f g]
     end
     return LocalLeftTensor(tmp * H.strength[], El.tag)
end

function _pushright(El::LocalLeftTensor{3}, A::AdjointMPSTensor{3}, H::LocalOperator{2,2}, B::MPSTensor{3}; kwargs...)
     if numout(A) == 1
          @tensor allocator = ManualAllocator() tmp[d; g h] := ((A.A[d a e] * El.A[a b c]) * H.A[b e f g]) * B.A[c f h]
     else
          @tensor allocator = ManualAllocator() tmp[d; g h] := ((A.A[e d a] * El.A[a b c]) * H.A[b e f g]) * B.A[c f h]
     end
     return LocalLeftTensor(tmp * H.strength[], (El.tag[1], H.tag[2][2], El.tag[3]))
end


# ========================= MPO ===========================
# TODO test performance
function _pushright(El::LocalLeftTensor{2}, A::AdjointMPSTensor{4}, B::MPSTensor{4}; kwargs...)
     @tensor allocator = ManualAllocator() tmp[f; e] := (El.A[a b] * A.A[d f a c]) * B.A[b c d e]
     return LocalLeftTensor(tmp, El.tag)
end

function _pushright(El::LocalLeftTensor{2}, A::AdjointMPSTensor{4}, H::IdentityOperator, B::MPSTensor{4}; kwargs...)
     return rmul!(_pushright(El, A, B), H.strength[])
end

function _pushright(El::LocalLeftTensor{2}, A::AdjointMPSTensor{4}, H::LocalOperator{1,1}, B::MPSTensor{4}; kwargs...)
     @tensor allocator = ManualAllocator() tmp[f; e] := ((El.A[a b] * A.A[d f a g]) * H.A[g c]) * B.A[b c d e]
     return LocalLeftTensor(rmul!(tmp, H.strength[]), El.tag)
end

function _pushright(El::LocalLeftTensor{2}, A::AdjointMPSTensor{4}, H::LocalOperator{1,2}, B::MPSTensor{4}; kwargs...)
     @tensor allocator = ManualAllocator() tmp[f; h e] := ((El.A[a b] * A.A[d f a g]) * H.A[g c h]) * B.A[b c d e]
     return LocalLeftTensor(rmul!(tmp, H.strength[]), (El.tag[1], H.tag[2][2], El.tag[2]))
end

function _pushright(El::LocalLeftTensor{3}, A::AdjointMPSTensor{4}, H::IdentityOperator, B::MPSTensor{4}; kwargs...)
     @tensor allocator = ManualAllocator() tmp[f; h e] := (El.A[a h b] * A.A[d f a c]) * B.A[b c d e]
     return LocalLeftTensor(rmul!(tmp, H.strength[]), El.tag)
end

function _pushright(El::LocalLeftTensor{3}, A::AdjointMPSTensor{4}, H::LocalOperator{1,1}, B::MPSTensor{4}; kwargs...)
     @tensor allocator = ManualAllocator() tmp[f; h e] := ((El.A[a h b] * A.A[d f a g]) * H.A[g c]) * B.A[b c d e]
     return LocalLeftTensor(rmul!(tmp, H.strength[]), El.tag)
end

function _pushright(El::LocalLeftTensor{3}, A::AdjointMPSTensor{4}, H::LocalOperator{2,1}, B::MPSTensor{4}; kwargs...)
     @tensor allocator = ManualAllocator() tmp[f; e] := ((El.A[a h b] * A.A[d f a g]) * H.A[h g c]) * B.A[b c d e]
     return LocalLeftTensor(rmul!(tmp, H.strength[]), (El.tag[1], El.tag[3]))
end

function _pushright(El::LocalLeftTensor{3}, A::AdjointMPSTensor{4}, H::LocalOperator{2,2}, B::MPSTensor{4}; kwargs...)

     @tensor allocator = ManualAllocator() tmp[f; i e] := ((El.A[a h b] * A.A[d f a g]) * H.A[h g c i]) * B.A[b c d e]
     return LocalLeftTensor(rmul!(tmp, H.strength[]), (El.tag[1], H.tag[2][2], El.tag[3]))
end
