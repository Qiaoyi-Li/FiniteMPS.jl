# some functions in TensorKit use multi-threading to parallelize the computation however seem to conflict with the high-level multi-threading implementations
function _add_general_kernel_sequential!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
	if iszero(β)
		zerovector!(tdst)
	elseif β != 1
		scale!(tdst, β)
	end
	for (f₁, f₂) in fusiontrees(tsrc)
		for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
			TensorOperations.tensoradd!(tdst[f₁′, f₂′], tsrc[f₁, f₂], p, false, α * coeff, One(), backend...)
		end
	end
	return nothing
end
function _add_abelian_kernel_sequential!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)

	for (f₁, f₂) in fusiontrees(tsrc)
		TensorKit._add_abelian_block!(tdst, tsrc, p, fusiontreetransform, f₁, f₂, α, β, backend...)
	end
	return tdst
end

TensorKit._add_abelian_kernel!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p, fusiontreetransform, α, β, backend...) = _add_abelian_kernel_sequential!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)

TensorKit._add_general_kernel!(tdst::AbstractTensorMap, tsrc::AbstractTensorMap, p, fusiontreetransform, α, β, backend...) = _add_general_kernel_sequential!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)

# multi-threading mul
using TensorKit: hasblock
function _mul_atomic!(tC::AbstractTensorMap,
	tA::AbstractTensorMap,
	tB::AbstractTensorMap, α::Number = 1.0, β::Number = 0.0)
	TensorKit.compose(space(tA), space(tB)) == space(tC) ||
		throw(SpaceMismatch(lazy"$(space(tC)) ≠ $(space(tA)) * $(space(tB))"))

	ntasks = get_num_threads_mul()
	if ntasks == 1 || !isa(blocksectors(tC), AbstractVector) || length(blocksectors(tC)) == 1
		return _mul_serial!(tC, tA, tB, α, β)
	else
		# sort sectors by size
		function blockcost(c)
			if hasblock(tA, c)
				return size(block(tA, c), 1) * size(block(tA, c), 2) * size(block(tB, c), 2)
			else
				return size(block(tC, c), 1) * size(block(tC, c), 2)
			end
		end
		sortedsectors = sort(blocksectors(tC); by = blockcost, rev = true)

		idx = Threads.Atomic{Int64}(1)
		Threads.@sync for _ in 1:ntasks
			t = Threads.@spawn while true
				i = Threads.atomic_add!(idx, 1)
				i > length(sortedsectors) && break

				c = sortedsectors[i]
				if hasblock(tA, c)
					mul!(block(tC, c),
						block(tA, c),
						block(tB, c),
						α, β)
				elseif β != one(β)
					rmul!(block(tC, c), β)
				end
			end
			errormonitor(t)
		end
	end

	return tC
end

function _mul_serial!(tC::AbstractTensorMap,
	tA::AbstractTensorMap,
	tB::AbstractTensorMap, α = 1.0, β = 0.0)
	# copy from TensorKit.mul!

	iterC = blocks(tC)
	iterA = blocks(tA)
	iterB = blocks(tB)
	nextA = iterate(iterA)
	nextB = iterate(iterB)
	nextC = iterate(iterC)
	while !isnothing(nextC)
		(cC, C), stateC = nextC
		if !isnothing(nextA) && !isnothing(nextB)
			(cA, A), stateA = nextA
			(cB, B), stateB = nextB
			if cA == cC && cB == cC
				mul!(C, A, B, α, β)
				nextA = iterate(iterA, stateA)
				nextB = iterate(iterB, stateB)
				nextC = iterate(iterC, stateC)
			elseif cA < cC
				nextA = iterate(iterA, stateA)
			elseif cB < cC
				nextB = iterate(iterB, stateB)
			else
				if β != one(β)
					rmul!(C, β)
				end
				nextC = iterate(iterC, stateC)
			end
		else
			if β != one(β)
				rmul!(C, β)
			end
			nextC = iterate(iterC, stateC)
		end
	end
	return tC
end

function TensorKit.mul!(tC::TensorMap{TC, <:GradedSpace}, tA::TensorMap{TA, <:GradedSpace}, tB::TensorMap{TB, <:GradedSpace}, α::Number, β::Number) where {TA <: Union{Float64, ComplexF64}, TB <: Union{Float64, ComplexF64}, TC <: Union{Float64, ComplexF64}}
	return _mul_atomic!(tC, tA, tB, α, β)
end

# multi-threading svd
using TensorKit: SectorDict, MatrixAlgebra
function _compute_svddata_threads!(t::TensorMap, alg::Union{SVD, SDD})

	I = sectortype(t)
	ntasks = get_num_threads_svd()
	if ntasks == 1 || !isa(blocksectors(t), AbstractVector) || length(blocksectors(t)) == 1
		for (c, b) in blocks(t)
			U, Σ, V = MatrixAlgebra.svd!(b, alg)
			SVDData[c] = (U, Σ, V)
			dims[c] = length(Σ)
		end
	else
		# pre allocate
		SVDData = SectorDict(map(blocks(t)) do (c, b)
			sz = size(b)
			U = similar(b, (sz[1], minimum(sz)))
			Σ = similar(b, real(scalartype(b)), minimum(sz))
			V = similar(b, (minimum(sz), sz[2]))
			return c => (U, Σ, V)
		end)
		dims = SectorDict{I, Int}(c => minimum(size(b)) for (c, b) in blocks(t))

		# sort sectors by size
		function blockcost(c)
			b = block(t, c)
			return min(size(b, 1)^2 * size(b, 2), size(b, 1) * size(b, 2)^2)
		end
		perms = sortperm(blocksectors(t); by = blockcost, rev = true)
		sortedsectors = blocksectors(t)[perms]

		Threads.@threads :greedy for c in sortedsectors
			inplace_svd!(SVDData[c][1], SVDData[c][2], SVDData[c][3], block(t, c), alg)
		end
	end
	return SVDData, dims
end

TensorKit._compute_svddata!(t::TensorMap{T, <:GradedSpace}, alg::Union{SVD, SDD}) where {T <: Union{Float64, ComplexF64}} = _compute_svddata_threads!(t, alg)

# multi-threading eigh
# TODO: implement in-place eig
using TensorKit: similarstoragetype, Diagonal
function _eigh_atomic!(t::TensorMap)
	InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:eigh!)
	domain(t) == codomain(t) ||
		throw(SpaceMismatch("`eigh!` requires domain and codomain to be the same"))

	S = spacetype(t)
	I = sectortype(t)
	A = storagetype(t)
	Ar = similarstoragetype(t, real(scalartype(t)))

	ntasks = get_num_threads_eig()
	if ntasks == 1 || !isa(blocksectors(t), AbstractVector) ||
	   length(blocksectors(t)) == 1
		Ddata = SectorDict{I, Ar}()
		Vdata = SectorDict{I, A}()
		dims = SectorDict{I, Int}()
		for (c, b) in blocks(t)
			values, vectors = MatrixAlgebra.eigh!(b)
			d = length(values)
			Ddata[c] = copyto!(similar(values, (d, d)), Diagonal(values))
			Vdata[c] = vectors
			dims[c] = d
		end

	else
		# preallocate
		_, b = first(blocks(t))
		emptymat = similar(b, (0, 0))
		Ddata = SectorDict{I, Ar}(c => emptymat for c in blocksectors(t))
		Vdata = SectorDict{I, A}(c => emptymat for c in blocksectors(t))
		dims = SectorDict{I, Int}(c => 0 for c in blocksectors(t))

		# sort sectors by size
		function blockcost(c)
			return size(block(t, c), 1)
		end
		perms = sortperm(blocksectors(t); by = blockcost, rev = true)
		sortedsectors = blocksectors(t)[perms]

		idx = Threads.Atomic{Int64}(1)
		Threads.@sync for _ in 1:ntasks
			Threads.@spawn while true
				i = Threads.atomic_add!(idx, 1)
				i > length(sortedsectors) && break

				c = sortedsectors[i]
				values, vectors = MatrixAlgebra.eigh!(block(t, c))
				d = length(values)
				Ddata[c] = copyto!(similar(values, (d, d)), Diagonal(values))
				Vdata[c] = vectors
				dims[c] = d
			end
		end
	end

	if length(domain(t)) == 1
		W = domain(t)[1]
	else
		W = S(dims)
	end
	return TensorMap(Ddata, W ← W), TensorMap(Vdata, domain(t) ← W)
end

TensorKit.eigh!(t::TensorMap{<:GradedSpace}) = _eigh_atomic!(t)
