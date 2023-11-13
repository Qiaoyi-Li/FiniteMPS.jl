"""
     oplusEmbed(A::AbstractTensorMap,
          B::AbstractTensorMap,
          idx::Int64) -> EmbA::TensorMap, EmbB::TensorMap

Return the 2 embedding maps (from `A` and `B`) to the direct sum space (or their adjoint) corresponding to `idx`.
"""
function oplusEmbed(A::AbstractTensorMap{F}, B::AbstractTensorMap{F}, idx::Int64) where {F<:GradedSpace}
     
     @assert rank(A) == rank(B)

     rA = rank(A, 1)
     if idx ≤ rA
          sumspace = codomain(A)[idx] ⊕ codomain(B)[idx]
          EmbA = isometry(sumspace, codomain(A)[idx])
          EmbB = isometry(sumspace, codomain(B)[idx])
          # adapt the data in EmbB, note dataB is a pointer, changing dataB will also change EmbB 
          dataB = data(EmbB)
          for i in eachindex(dataB)
               sz = size(dataB[i])
               @assert sz[1] ≥ sz[2]
               dataB[i] = vcat(zeros(scalartype(B), sz[1] - sz[2], sz[2]), Matrix{scalartype(B)}(I, sz[2], sz[2]))
          end
     else
          sumspace = domain(A)[idx-rA] ⊕ domain(B)[idx-rA]
          EmbA = isometry(sumspace, domain(A)[idx-rA])'
          EmbB = isometry(sumspace, domain(B)[idx-rA])'
          # adapt the data in EmbB, note dataB is a pointer, changing dataB will also change EmbB 
          dataB = data(EmbB)
          for i in eachindex(dataB)
               sz = size(dataB[i])
               @assert sz[1] ≥ sz[2]
               dataB[i] = vcat(zeros(scalartype(B), sz[1] - sz[2], sz[2]), Matrix{scalartype(B)}(I, sz[2], sz[2]))
          end
     end
     return EmbA, EmbB
end

# function oplusEmbed(A::AbstractTensorMap{F}, B::AbstractTensorMap{F}, idx::Int64) where {F<:Union{CartesianSpace,ComplexSpace}}
#      # return the 2 embedding maps (from A and B) to the direct sum space (or their adjoint) of given idx

#      @assert rank(A) == rank(B)

#      rA = rank(A, 1)
#      if idx ≤ rA
#           sumspace = codomain(A)[idx] ⊕ codomain(B)[idx]
#           EmbA = isometry(sumspace, codomain(A)[idx])
#           EmbB = isometry(sumspace, codomain(B)[idx])
#           # data is just a matrix for pure tensor in ℝ or ℂ 
#           dataB = data(EmbB)
#           sz = size(dataB)
#           @assert sz[1] ≥ sz[2]
#           dataB[:] = vcat(zeros(eltype(B), sz[1] - sz[2], sz[2]), Matrix{eltype(B)}(I, sz[2], sz[2]))[:]

#      else
#           sumspace = domain(A)[idx-rA] ⊕ domain(B)[idx-rA]
#           EmbA = isometry(sumspace, domain(A)[idx-rA])'
#           EmbB = isometry(sumspace, domain(B)[idx-rA])'
#           # data is just a matrix for pure tensor in ℝ or ℂ
#           dataB = data(EmbB)
#           sz = size(dataB)
#           @assert sz[1] ≥ sz[2]
#           # "[:]" cannot be ignored to make sure dataB === data(EmbB)
#           dataB[:] = vcat(zeros(eltype(B), sz[1] - sz[2], sz[2]), Matrix{eltype(B)}(I, sz[2], sz[2]))[:]
#      end
#      return EmbA, EmbB
# end