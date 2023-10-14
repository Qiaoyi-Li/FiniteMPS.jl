# replace some TensorKit functions to avoid conflict of multi-threading
function TensorKit._add_general_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
    if iszero(β)
        zerovector!(tdst)
    elseif β != 1
        scale!(tdst, β)
    end

    for (f₁, f₂) in fusiontrees(tsrc)
        for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
            TensorOperations.tensoradd!(tdst[f₁′, f₂′], p, tsrc[f₁, f₂], :N, α * coeff, true,
                backend...)
        end
    end

    return nothing
end


function TensorKit._add_abelian_kernel!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)

    for (f₁, f₂) in fusiontrees(tsrc)
        TensorKit._add_abelian_block!(tdst, tsrc, p, fusiontreetransform, f₁, f₂, α, β, backend...)
    end

    return tdst
end

function TensorKit.mul!(tC::AbstractTensorMap,
    tA::AbstractTensorMap,
    tB::AbstractTensorMap, α=true, β=false)

    if !(codomain(tC) == codomain(tA) && domain(tC) == domain(tB) &&
         domain(tA) == codomain(tB))
        throw(SpaceMismatch("$(space(tC)) ≠ $(space(tA)) * $(space(tB))"))
    end

    # parellel this function only in multi-processing mode
    if get_num_workers() > 1 && get_num_threads_julia() > 1

        @floop GlobalThreadsExecutors for c in filter(c -> TensorKit.hasblock(tA, c), blocksectors(tC))
            mul!(TensorKit.StridedView(block(tC, c)), TensorKit.StridedView(block(tA, c)), TensorKit.StridedView(block(tB, c)), α, β)
        end

        if β != one(β)
            @floop GlobalThreadsExecutors for c in filter(c -> !TensorKit.hasblock(tA, c), blocksectors(tC))
                rmul!(block(tC, c), β)
            end
        end

    else
        for c in blocksectors(tC)
            if TensorKit.hasblock(tA, c) # then also tB should have such a block
                A = block(tA, c)
                B = block(tB, c)
                C = block(tC, c)
                mul!(TensorKit.StridedView(C), TensorKit.StridedView(A), TensorKit.StridedView(B), α, β)
            elseif β != one(β)
                rmul!(block(tC, c), β)
            end
        end
    end
    return tC
end
