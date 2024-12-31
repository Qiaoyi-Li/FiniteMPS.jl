"""
     oplusEmbed(lsV::Vector{<:GradedSpace};
          rev::Bool=false) -> lsEmbed::Vector{<:AbstractTensorMap}

Return the embedding maps from vectors in `lsV` to their direct sum space, with the same order as `lsV`. If `rev == true`, return the submersions from the direct sum space to the vectors instead.

     oplusEmbed(A::AbstractTensorMap,
          B::AbstractTensorMap,
          idx::Int64) -> EmbA::TensorMap, EmbB::TensorMap

Return the 2 embedding maps (from `A` and `B`) to the direct sum space (or their adjoint) corresponding to `idx`.
"""
function oplusEmbed(lsV::Vector{<:GradedSpace}; rev::Bool=false)

     V_oplus = ⊕(lsV...)
     dims_count = TensorKit.SectorDict{sectortype(V_oplus),Int64}()
     for c in sectors(V_oplus)
          dims_count[c] = 0
     end
     lsEmbed = map(lsV) do V
          rev ? TensorMap(zeros, V, V_oplus) : TensorMap(zeros, V_oplus, V)
     end

     for (V, Embed) in zip(lsV, lsEmbed)
          for c in sectors(V)
               d = dim(V, c)
               if rev
                    block(Embed, c)[:, dims_count[c]+1:dims_count[c]+d] = Matrix(I, d, d)
               else
                    block(Embed, c)[dims_count[c]+1:dims_count[c]+d, :] = Matrix(I, d, d)
               end
               dims_count[c] += d
          end
     end

     return lsEmbed
end
oplusEmbed(lsV::GradedSpace...; kwargs...) = oplusEmbed([lsV...,]; kwargs...)

function oplusEmbed(lsV::Vector{T}; rev::Bool=false) where T <: Union{CartesianSpace,ComplexSpace}

    V_oplus = ⊕(lsV...)
    dims_count = 0
    lsEmbed = map(lsV) do V
         rev ? TensorMap(zeros, V, V_oplus) : TensorMap(zeros, V_oplus, V)
    end

    for (V, Embed) in zip(lsV, lsEmbed)
        d = dim(V)
        if rev
            block(Embed, Trivial())[:, dims_count+1:dims_count+d] = Matrix(I, d, d)
        else
               block(Embed, Trivial())[dims_count+1:dims_count+d, :] = Matrix(I, d, d)
        end
        dims_count += d
    end

    return lsEmbed
end
oplusEmbed(lsV::T...; kwargs...) where T <: Union{CartesianSpace,ComplexSpace} = oplusEmbed([lsV...,]; kwargs...)

function oplusEmbed(A::AbstractTensorMap, B::AbstractTensorMap, idx::Int64)

     @assert rank(A) == rank(B)
     T = promote_type(eltype(A), eltype(B))

     rA = rank(A, 1)
     if idx ≤ rA
          sumspace = codomain(A)[idx] ⊕ codomain(B)[idx]
          EmbA = isometry(T, sumspace, codomain(A)[idx])
          EmbB = TensorMap(zeros, T, sumspace, codomain(B)[idx])
          for (c, b) in blocks(EmbB)
               sz = size(b)
               @assert sz[1] ≥ sz[2]
               b[sz[1] - sz[2] + 1:end, :] .= Matrix{T}(I, sz[2], sz[2])
          end
     else
          sumspace = domain(A)[idx-rA] ⊕ domain(B)[idx-rA]
          EmbA = isometry(T, sumspace, domain(A)[idx-rA])'
          EmbB = TensorMap(zeros, T, sumspace, domain(B)[idx-rA])'
          for (c, b) in blocks(EmbB)
               sz = size(b)
               @assert sz[1] ≤ sz[2]
               b[:, sz[2] - sz[1] + 1:end] .= Matrix{T}(I, sz[1], sz[1])
          end
     end
     return EmbA, EmbB
end
