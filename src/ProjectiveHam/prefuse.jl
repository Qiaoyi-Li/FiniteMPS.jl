"""
     struct PreFuseProjectiveHamiltonian{N, Tl, Tr} <: AbstractProjectiveHamiltonian 
          El::Tl
          Er::Tr
          si::Vector{Int64}
          E₀::Float64
     end

Prefused `N`-site projective Hamiltonian. Note `El` and `Er` can be a original environment tensor or a prefused one, depending on `N`. If `N == 1`, only one of them will be prefused.
"""
struct PreFuseProjectiveHamiltonian{N,Tl,Tr} <: AbstractProjectiveHamiltonian
     El::Tl
     Er::Tr
     si::Vector{Int64}
     E₀::Float64
     function PreFuseProjectiveHamiltonian(El::Tl,
          Er::Tr,
          si::Vector{Int64},
          E₀::Float64=0.0) where {Tl<:Union{SparseLeftTensor,SparseLeftPreFuseTensor},Tr}
          @assert length(si) == 2
          N = si[2] - si[1] + 1
          return new{N,Tl,Tr}(El, Er, si, E₀)
     end
end

function _prefuse(PH::SparseProjectiveHamiltonian{1})
     El = _prefuse(PH.El, PH.H[1], PH.validIdx)
     idx = findall(x -> isassigned(El, x), eachindex(El))
     return PreFuseProjectiveHamiltonian(El[idx], PH.Er[idx], PH.si, PH.E₀)
end

function _prefuse(lsEl::SparseLeftTensor, Hl::SparseMPOTensor, validIdx::Vector{Tuple})
     sz = size(Hl)
     El_next = SparseLeftPreFuseTensor(undef, sz[2])
     if get_num_workers() > 1
          # TODO

     else
         
          Lock = Threads.ReentrantLock()
          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true
                    idx_t = Threads.atomic_add!(idx, 1)
                    idx_t > length(validIdx) && break
                    (i, j) = validIdx[idx_t]
                    El = _prefuse(lsEl[i], Hl[i, j])

                    lock(Lock)
                    try
                         if !isassigned(El_next, j)
                              El_next[j] = El
                         else
                              axpy!(true, El, El_next[j])
                         end
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

function _prefuse(El::LocalLeftTensor{2}, H::IdentityOperator)
     pspace = getPhysSpace(H)
     Id = isometry(pspace, pspace)
     @tensor tmp[a d; c e] := H.strength * El.A[a c] * Id[d e]
     return tmp
end
function _prefuse(El::LocalLeftTensor{2}, H::LocalOperator{1,1})
     @tensor tmp[a d; c e] := H.strength * El.A[a c] * H.A[d e]
     return tmp
end
function _prefuse(El::LocalLeftTensor{2}, H::LocalOperator{1, 2})
     @tensor tmp[a d f; c e] := H.strength * El.A[a c] * H.A[d e f]
     return tmp
end
function _prefuse(El::LocalLeftTensor{3}, H::LocalOperator{2, 1})
     @tensor tmp[a d; c e] := H.strength * El.A[a b c] * H.A[b d e]
     return tmp
end
function _prefuse(El::LocalLeftTensor{3}, H::LocalOperator{1, 1})
     @tensor tmp[a d b; c e] := H.strength * El.A[a b c] * H.A[d e]
     return tmp
end
