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

        @floop GlobalThreadsExecutor for c in filter(c -> TensorKit.hasblock(tA, c), blocksectors(tC))
            mul!(TensorKit.StridedView(block(tC, c)), TensorKit.StridedView(block(tA, c)), TensorKit.StridedView(block(tB, c)), α, β)
        end

        if β != one(β)
            @floop GlobalThreadsExecutor for c in filter(c -> !TensorKit.hasblock(tA, c), blocksectors(tC))
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

# use multi-threading to accelerate svd via dispatching blocks to different threads
function TensorKit.tsvd!(t::TensorMap;
    trunc::TruncationScheme=TensorKit.NoTruncation(),
    p::Real=2,
    alg::Union{SVD,SDD}=SDD())
    InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:tsvd!)
    S = spacetype(t)
    I = sectortype(t)
    A = storagetype(t)
    Ar = TensorKit.similarstoragetype(t, real(scalartype(t)))
    Udata = TensorKit.SectorDict{I,A}()
    Σmdata = TensorKit.SectorDict{I,Ar}() # this will contain the singular values as matrix
    Vdata = TensorKit.SectorDict{I,A}()
    dims = TensorKit.SectorDict{sectortype(t),Int}()
    if isempty(blocksectors(t))
        W = S(dims)
        truncerr = zero(real(scalartype(t)))
        return TensorMap(Udata, codomain(t) ← W), TensorMap(Σmdata, W ← W),
        TensorMap(Vdata, W ← domain(t)), truncerr
    end

    if Threads.nthreads() > 1

        lsc = collect(keys(blocks(t)))
        lsU = Vector{A}(undef, length(lsc))
        lsΣ = Vector{Vector{real(scalartype(t))}}(undef, length(lsc))
        lsV = Vector{A}(undef, length(lsc))

        @floop GlobalThreadsExecutor for i in 1:length(lsc)
            U, Σ, V = TensorKit._svd!(blocks(t)[lsc[i]], alg)
            lsU[i] = U
            lsΣ[i] = Σ
            lsV[i] = V
        end

        for i in 1:length(lsc)
            c = lsc[i]
            Udata[c] = lsU[i]
            Vdata[c] = lsV[i]
            if @isdefined Σdata # cannot easily infer the type of Σ, so use this construction
                Σdata[c] = lsΣ[i]
            else
                Σdata = TensorKit.SectorDict(c => lsΣ[i])
            end
            dims[c] = length(lsΣ[i])
        end

    else
        for (c, b) in blocks(t)
            U, Σ, V = TensorKit._svd!(b, alg)
            Udata[c] = U
            Vdata[c] = V
            if @isdefined Σdata # cannot easily infer the type of Σ, so use this construction
                Σdata[c] = Σ
            else
                Σdata = TensorKit.SectorDict(c => Σ)
            end
            dims[c] = length(Σ)
        end
    end
    if !isa(trunc, TensorKit.NoTruncation)
        Σdata, truncerr = TensorKit._truncate!(Σdata, trunc, p)
        truncdims = TensorKit.SectorDict{I,Int}()
        for c in blocksectors(t)
            truncdim = length(Σdata[c])
            if truncdim != 0
                truncdims[c] = truncdim
                if truncdim != dims[c]
                    Udata[c] = Udata[c][:, 1:truncdim]
                    Vdata[c] = Vdata[c][1:truncdim, :]
                end
            else
                delete!(Udata, c)
                delete!(Vdata, c)
                delete!(Σdata, c)
            end
        end
        dims = truncdims
        W = S(dims)
    else
        W = S(dims)
        if length(domain(t)) == 1 && domain(t)[1] ≅ W
            W = domain(t)[1]
        elseif length(codomain(t)) == 1 && codomain(t)[1] ≅ W
            W = codomain(t)[1]
        end
        truncerr = abs(zero(scalartype(t)))
    end
    for (c, Σ) in Σdata
        Σmdata[c] = copyto!(similar(Σ, length(Σ), length(Σ)), TensorKit.Diagonal(Σ))
    end
    return TensorMap(Udata, codomain(t) ← W), TensorMap(Σmdata, W ← W),
    TensorMap(Vdata, W ← domain(t)), truncerr
end
