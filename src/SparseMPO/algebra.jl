# TODO
# function adjoint(A::SparseMPO)
#      B = Vector{SparseMPOTensor}(undef, length(A))

#      collect_Op = Vector{AbstractTensorMap}(undef, 0)
#      ref = Vector{Tuple{Int64,Int64}}(undef, 0)
#      for i in eachindex(B)
#           B[i] = similar(A[i])
#           for idx in eachindex(A[i])
#                if ismissing(A[i][idx])
#                     B[i][idx] = missing
#                     continue
#                end

#                a = findfirst(x -> x === A[i][idx], collect_Op)
#                if isnothing(a)
#                     B[i][idx] = permute(A[i][idx]', (3, 1), (4, 2))
#                     push!(collect_Op, A[i][idx])
#                     push!(ref, (i, idx))
#                else
#                     B[i][idx] = B[ref[a][1]][ref[a][2]]
#                end
#           end
#      end
#      return SparseMPO(B)
# end

# function LinearCombination(A::SparseMPO, cA::Number, B::SparseMPO, cB::Number)
#      # return C = cA*A + cB*B
#      L = length(A)
#      @assert length(B) == L

#      @floop begin
#           WorkStealingEx()
#           C = Vector{SparseMPOTensor}(undef, L)
#           for si in 1:L
#                sz_A = size(A[si])
#                sz_B = size(B[si])
#                C[si] = Matrix{Union{Missing, AbstractTensorMap}}(missing, sz_A[1]+sz_B[1], sz_A[2] + sz_B[2])
#                C[si][1:sz_A[1], 1:sz_A[2]] = si == 1 ? cA*A[si][:] : A[si][:]
#                C[si][(sz_A[1] + 1): end, (sz_A[2] + 1): end] = si == 1 ? cB*B[si][:] : B[si][:]
#           end
#      end

#      return SparseMPO(C)

# end