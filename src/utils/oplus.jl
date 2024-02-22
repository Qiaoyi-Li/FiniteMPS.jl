"""
     oplusEmbed(lsV::Vector{<:GradedSpace};
          reverse::Bool=false) -> lsEmbed::Vector{<:AbstractTensorMap}

Return the embedding maps from vectors in `lsV` to their direct sum space, with the same order as `lsV`. If `reverse == true`, return the submersions from the direct sum space to the vectors instead.

     oplusEmbed(A::AbstractTensorMap,
          B::AbstractTensorMap,
          idx::Int64) -> EmbA::TensorMap, EmbB::TensorMap

Return the 2 embedding maps (from `A` and `B`) to the direct sum space (or their adjoint) corresponding to `idx`.
"""
function oplusEmbed(lsV::Vector{<:GradedSpace}; reverse::Bool=false)

     V_oplus = ⊕(lsV...)
     dims_count = TensorKit.SectorDict{sectortype(V_oplus),Int64}()
     for c in sectors(V_oplus)
          dims_count[c] = 0
     end
     lsEmbed = map(lsV) do V
          reverse ? TensorMap(zeros, V, V_oplus) : TensorMap(zeros, V_oplus, V)
     end

     for (V, Embed) in zip(lsV, lsEmbed)
          for c in sectors(V)
               d = dim(V, c)
               if reverse
                    data(Embed)[c][:, dims_count[c]+1:dims_count[c]+d] = Matrix(I, d, d)
               else
                    data(Embed)[c][dims_count[c]+1:dims_count[c]+d, :] = Matrix(I, d, d)
               end
               dims_count[c] += d
          end
     end

     return lsEmbed
end
oplusEmbed(lsV::GradedSpace...; kwargs...) = oplusEmbed([lsV...,]; kwargs...)

function oplusEmbed(lsV::Vector{T}; reverse::Bool=false) where T <: Union{CartesianSpace,ComplexSpace}

    V_oplus = ⊕(lsV...)
    dims_count = 0
    lsEmbed = map(lsV) do V
         reverse ? TensorMap(zeros, V, V_oplus) : TensorMap(zeros, V_oplus, V)
    end

    for (V, Embed) in zip(lsV, lsEmbed)
        d = dim(V)
        if reverse
            data(Embed)[1][:, dims_count+1:dims_count+d] = Matrix(I, d, d)
        else
            data(Embed)[1][dims_count+1:dims_count+d, :] = Matrix(I, d, d)
        end
        dims_count += d
    end

    return lsEmbed
end
oplusEmbed(lsV::T...; kwargs...) where T <: Union{CartesianSpace,ComplexSpace} = oplusEmbed([lsV...,]; kwargs...)

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

function oplusEmbed(A::AbstractTensorMap{F}, B::AbstractTensorMap{F}, idx::Int64) where {F<:Union{CartesianSpace,ComplexSpace}}
     # return the 2 embedding maps (from A and B) to the direct sum space (or their adjoint) of given idx

     @assert rank(A) == rank(B)

     rA = rank(A, 1)

     if idx ≤ rA
          sumspace = codomain(A)[idx] ⊕ codomain(B)[idx]
          EmbA = isometry(sumspace, codomain(A)[idx])
          EmbB = isometry(sumspace, codomain(B)[idx])
          # data is just a matrix for pure tensor in ℝ or ℂ
          dataB = data(EmbB)[1]
          sz = size(dataB)
          @assert sz[1] ≥ sz[2]
          dataB[:] = vcat(zeros(eltype(B), sz[1] - sz[2], sz[2]), Matrix{eltype(B)}(I, sz[2], sz[2]))[:]

     else
          sumspace = domain(A)[idx-rA] ⊕ domain(B)[idx-rA]
          EmbA = isometry(sumspace, domain(A)[idx-rA])'
          EmbB = isometry(sumspace, domain(B)[idx-rA])'
          # data is just a matrix for pure tensor in ℝ or ℂ
          dataB = data(EmbB)[1]
          sz = size(dataB)
          @assert sz[1] ≥ sz[2]
          # "[:]" cannot be ignored to make sure dataB === data(EmbB)
          dataB[:] = vcat(zeros(eltype(B), sz[1] - sz[2], sz[2]), Matrix{eltype(B)}(I, sz[2], sz[2]))[:]
     end
     return EmbA, EmbB
end
