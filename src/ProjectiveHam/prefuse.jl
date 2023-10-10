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