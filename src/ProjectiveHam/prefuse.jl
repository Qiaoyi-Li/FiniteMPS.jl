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

function prefuse(PH::SparseProjectiveHamiltonian{1})
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
          El_fetch = SparseLeftPreFuseTensor(undef, length(validIdx))
          @floop GlobalThreadsExecutor for (n, (i, j)) in enumerate(validIdx)
               El_fetch[n] = _prefuse(lsEl[i], Hl[i, j])
          end
          for (n, (_, j)) in enumerate(validIdx)
               if !isassigned(El_next, j)
                    El_next[j] = El_fetch[n]
               else
                    axpy!(true, El_fetch[n], El_next[j])
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




# 2023.10.01 qyli@itp prefuse may not be needed for TensorKit
# function _prefuse2(PH::SparseProjectiveHamiltonian{2})

#      sz = (size(PH.H[1], 1), size(PH.H[1], 2), size(PH.H[2], 2))
#      El = SparseEnvironmentTensor(undef, sz[2])
#      Er = SparseEnvironmentTensor(undef, sz[2])

#      idxL = Tuple{Int64,Int64}[]
#      idxR = Tuple{Int64,Int64}[]
#      for (i, j, k) in PH.validIdx
#           if isempty(idxL) || idxL[end] != (i, j)
#                push!(idxL, (i, j))
#           end
#           if isempty(idxR) || idxR[end] != (j, k)
#                push!(idxR, (j, k))
#           end
#      end

#      lspace = domain(PH.El[findfirst(!ismissing, PH.El)])[end]
#      pspace = domain(PH.H[1][findfirst(!ismissing, PH.H[1])])[1]
#      rspace = domain(PH.Er[findfirst(!ismissing, PH.Er)])[end]
#      isoL = isometry(lspace ⊗ pspace, fuse(lspace, pspace))
#      isoR = isometry(rspace ⊗ pspace, fuse(rspace, pspace))

#      @sync begin
#           @spawn begin
#                @threads for j in 1:sz[2]
#                     @floop WorkStealingEx(; simd=true) for i in filter(x -> (x, j) in idxL, 1:sz[1])
#                          @tensor x1[h; f g] := ((PH.El[i][a, b, c] * isoL[c, e, g]) * PH.H[1][i, j][b, d, e, f]) * isoL'[h, a, d]
#                          @reduce() do (acc1 = missing; x1)
#                               acc1 += x1
#                          end
#                     end
#                     El[j] = acc1
#                end
#           end
#           @spawn begin
#                @threads for j in 1:sz[2]
#                     @floop WorkStealingEx(; simd=true) for k in filter(x -> (j, x) in idxR, 1:sz[3])
#                          @tensor x2[h; f e] := ((PH.Er[k][a, b, c] * isoR[c, d, e]) * PH.H[2][j, k][f, g, d, b]) * isoR'[h, a, g]
#                          @reduce() do (acc2 = missing; x2)
#                               acc2 += x2
#                          end
#                     end
#                     Er[j] = acc2
#                end
#           end
#      end

#      return El, Er, isoL, isoR

# end