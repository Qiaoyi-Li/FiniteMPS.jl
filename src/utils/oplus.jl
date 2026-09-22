"""
     oplusEmbed(lsV::Vector{<:ElementarySpace};
          rev::Bool=false) -> lsEmbed::Vector{<:AbstractTensorMap}

Return the embedding maps from vectors in `lsV` to their direct sum space, with the same order as `lsV`. If `rev == true`, return the submersions from the direct sum space to the vectors instead.
"""
function oplusEmbed(lsV::Vector{<:ElementarySpace}; rev::Bool=false)

     V_oplus = ⊕(lsV...)
     dims_count = Dict(c => 0 for c in sectors(V_oplus))
     lsEmbed = map(lsV) do V
          rev ? zeros(V, V_oplus) : zeros(V_oplus, V)
     end

     for (V, Embed) in zip(lsV, lsEmbed)
          for c in sectors(V)
               d = dim(V, c)
               b = block(Embed, c)
               offset = dims_count[c]
               for i in 1:d
                    if rev
                         b[i, offset+i] = 1
                    else
                         b[offset+i, i] = 1
                    end
               end
               dims_count[c] += d
          end
     end

     return lsEmbed
end
oplusEmbed(lsV::ElementarySpace...; kwargs...) = oplusEmbed([lsV...,]; kwargs...)
