# replace some TensorKit functions to avoid conflict of multi-threading
# TODO add more parameters to control the 3-layer nested multi-threading
function TensorKit._add_general_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
    if iszero(β)
        tdst = zerovector!(tdst)
    elseif β != 1
        tdst = scale!(tdst, β)
    end
    # if Threads.nthreads() > 1
    #     Threads.@sync for s₁ in sectors(codomain(tsrc)), s₂ in sectors(domain(tsrc))
    #         Threads.@spawn _add_nonabelian_sector!(tdst, tsrc, p, fusiontreetransform, s₁,
    #                                                s₂, α, β, backend...)
    #     end
    # else
    for (f₁, f₂) in fusiontrees(tsrc)
        for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
            TensorOperations.tensoradd!(tdst[f₁′, f₂′], p, tsrc[f₁, f₂], :N, α * coeff, true,
                backend...)
        end
    end
    # end
    return nothing
end

function TensorKit._add_abelian_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
    # if Threads.nthreads() > 1
    #     Threads.@sync for (f₁, f₂) in fusiontrees(tsrc)
    #         Threads.@spawn _add_abelian_block!(tdst, tsrc, p, fusiontreetransform,
    #                                            f₁, f₂, α, β, backend...)
    #     end
    # else
    for (f₁, f₂) in fusiontrees(tsrc)
        TensorKit._add_abelian_block!(tdst, tsrc, p, fusiontreetransform,
            f₁, f₂, α, β, backend...)
    end
    # end
    return tdst
end

# TODO maybe add multi-threading for tensor contraction is meaningful
# function LinearAlgebra.mul!(tC::AbstractTensorMap,
#     tA::AbstractTensorMap,
#     tB::AbstractTensorMap, α=true, β=false)
#     if !(codomain(tC) == codomain(tA) && domain(tC) == domain(tB) &&
#          domain(tA) == codomain(tB))
#         throw(SpaceMismatch("$(space(tC)) ≠ $(space(tA)) * $(space(tB))"))
#     end
#     for c in blocksectors(tC)
#         if hasblock(tA, c) # then also tB should have such a block
#             A = block(tA, c)
#             B = block(tB, c)
#             C = block(tC, c)
#             mul!(StridedView(C), StridedView(A), StridedView(B), α, β)
#         elseif β != one(β)
#             rmul!(block(tC, c), β)
#         end
#     end
#     return tC
# end