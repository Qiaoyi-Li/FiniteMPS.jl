"""
	 abstract type AbstractProjectiveHamiltonian
Abstract type of all projective Hamiltonian.
"""
abstract type AbstractProjectiveHamiltonian end

"""
	mutable struct SimpleProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
		El::LocalLeftTensor
	 	Er::LocalRightTensor
	 	H::NTuple{N, AbstractLocalOperator}
	 	cache::Vector{<:AbstractTensorMap}
	end

Concrete type of simple (i.e. `El`, `H` and `Er` define a single interaction term) `N`-site projective Hamiltonian.

# Fields
	El::LocalLeftTensor
	Er::LocalRightTensor
Left and right environment tensors.

	H::NTuple{N, AbstractLocalOperator}
The `N` local operators in Hamiltonian that define the interaction term.

	cache::Vector{<:AbstractTensorMap}
A cache to store the intermediate tensors when the projective Hamiltonian acts on a state, which will be freed when the object is finalized.

# Constructors
	SimpleProjectiveHamiltonian(El::LocalLeftTensor, Er::LocalRightTensor, H::AbstractLocalOperator...)
Construct a simple projective Hamiltonian from `El`, `Er` and `H`, where `N` is automatically deduced from the length of `H`.
"""
mutable struct SimpleProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
	El::LocalLeftTensor
	Er::LocalRightTensor
	H::NTuple{N, AbstractLocalOperator}
	cache::Vector{<:AbstractTensorMap}
	function SimpleProjectiveHamiltonian(
		El::LocalLeftTensor,
		Er::LocalRightTensor,
		H::AbstractLocalOperator...,
	)
		N = length(H)
		obj = new{N}(El, Er, Tuple(H), AbstractTensorMap[])

		# clean 
		finalizer(obj) do o
			for t in o.cache
				tensorfree!(t, ManualAllocator())
			end
			empty!(o.cache)
			return nothing
		end
		return obj
	end
end

"""
	mutable struct CompositeProjectiveHamiltonian{L} <: AbstractProjectiveHamiltonian
		PH::Vector{SimpleProjectiveHamiltonian{L}}
		E₀::Float64
	end

Concrete type of composite projective Hamiltonian, which is a collection of simple projective Hamiltonians.

# Fields
	PH::Vector{SimpleProjectiveHamiltonian{L}}
A vector that stores all contained simple projective Hamiltonians.

	E₀::Float64
The energy offset. The projective Hamiltonian actually acts on a state as `H - E₀`, usually used to avoid numerical unstableness in Krylov methods.

# Constructors
	CompositeProjectiveHamiltonian(El::SparseLeftTensor, Er::SparseRightTensor, H::NTuple{L, SparseMPOTensor}, E₀::Float64 = 0.0) 
Construct a composite projective Hamiltonian from the sparse left and right tensors `El`, `Er` and `H`, which directly provided by a `SparseEnvironment`.
"""
mutable struct CompositeProjectiveHamiltonian{L} <: AbstractProjectiveHamiltonian
	PH::Vector{SimpleProjectiveHamiltonian{L}}
	E₀::Float64
	function CompositeProjectiveHamiltonian(
		El::SparseLeftTensor,
		Er::SparseRightTensor,
		H::NTuple{L, SparseMPOTensor},
		E₀::Float64 = 0.0) where L

		validIdx = _countIntr(El, Er, H)

		lsPH = map(validIdx) do idx
			Hi = map(1:L) do i
				H[i][idx[i], idx[i+1]]
			end
			SimpleProjectiveHamiltonian(El[idx[1]], Er[idx[end]], Hi...)
		end


		obj = new{L}(lsPH, E₀)

		# finalizer 
		finalizer(obj) do o
			for PH in o.PH
				finalize(PH)
			end
			return nothing
		end

		return obj

	end
end

"""
	struct IdentityProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
		El::SimpleLeftTensor
		Er::SimpleRightTensor
		si::Vector{Int64}
	end

Special type to deal with the cases which satisfy ⟨Ψ₁|Id|Ψ₂⟩ == ⟨Ψ₁|Ψ₂⟩, thus the environment is a 2-layer simple one.

# Fields
	El::SimpleLeftTensor
	Er::SimpleRightTensor
Left and right environment tensors.

	si::Vector{Int64}
A length-`2` vector to label the starting and ending sites of the projective Hamiltonian.

# Constructors
	IdentityProjectiveHamiltonian(El::SimpleLeftTensor, Er::SimpleRightTensor, si::Vector{Int64})
Construct a projective Hamiltonian that corresponds to an identity operator, where the sites are deduced from `si`.
"""
struct IdentityProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
	El::SimpleLeftTensor
	Er::SimpleRightTensor
	si::Vector{Int64}
	function IdentityProjectiveHamiltonian(El::SimpleLeftTensor,
		Er::SimpleRightTensor,
		si::Vector{Int64})
		N = si[2] - si[1] + 1
		obj = new{N}(El, Er, si)
		return obj
	end
end

"""
	 struct SparseProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian  
		  El::SparseLeftTensor
		  Er::SparseRightTensor
		  H::NTuple{N, SparseMPOTensor}
		  si::Vector{Int64}
		  validIdx::Vector{Tuple}
		  E₀::Float64
	 end

`N`-site projective Hamiltonian, sparse version. Note we shift `H` to `H - E₀` to avoid numerical overflow.

Convention:
	  --               --       --                          --
	 |         |         |     |         |          |         |
	 El-- i -- H1 -- j --Er    El-- i -- H1 -- j -- H2 -- k --Er    ...
	 |         |         |     |         |          |         |
	  --               --       --                          --

`validIdx` stores all tuples `(i, j, ...)` which are valid, i.e. all `El[i]`, `H1[i, j]` and `Er[j]` are not `nothing` (`N == 1`). 
"""
struct SparseProjectiveHamiltonian{N} <: AbstractProjectiveHamiltonian
	El::SparseLeftTensor
	Er::SparseRightTensor
	H::NTuple{N, SparseMPOTensor}
	si::Vector{Int64}
	validIdx::Vector{Tuple}
	E₀::Float64
	function SparseProjectiveHamiltonian(El::SparseLeftTensor,
		Er::SparseRightTensor,
		H::NTuple{N, SparseMPOTensor},
		si::Vector{Int64},
		E₀::Float64 = 0.0,
	) where {N}
		validIdx = NTuple{N + 1, Int64}[]
		obj = new{N}(El, Er, H, si, validIdx, E₀)
		push!(obj.validIdx, _countIntr(obj)...)
		return obj
	end
end

"""
	 ProjHam(Env::SparseEnvironment, siL::Int64 [, siR::Int64 = siL]; E₀::Number = 0.0)

Generic constructor for N-site projective Hamiltonian, where `N = siR - siL + 1`.

	 ProjHam(Env::SimpleEnvironment, siL::Int64 [, siR::Int64 = siL])

Construct the special `IdentityProjectiveHamiltonian` from a simple environment.
"""
function ProjHam(Env::SparseEnvironment{L, 3, T}, siL::Int64, siR::Int64; E₀::Number = 0.0) where {L, T <: Tuple{AdjointMPS, SparseMPO, DenseMPS}}
	@assert 1 ≤ siL ≤ Env.Center[1] && Env.Center[2] ≤ siR ≤ L # make sure El and Er are valid
	N = siR - siL + 1
	@assert N ≥ 0
	return SparseProjectiveHamiltonian(Env.El[siL], Env.Er[siR], Tuple(Env[2][siL:siR]), [siL, siR], convert(Float64, E₀))
end
function ProjHam(Env::SimpleEnvironment{L, 2, T}, siL::Int64, siR::Int64) where {L, T <: Tuple{AdjointMPS, DenseMPS}}
	@assert siL ≤ Center(Env)[1] && siR ≥ Center(Env)[2] # make sure El and Er are valid
	return IdentityProjectiveHamiltonian(Env.El[siL], Env.Er[siR], [siL, siR])
end
ProjHam(Env::AbstractEnvironment, si::Int64; kwargs...) = ProjHam(Env, si, si; kwargs...)

function show(io::IO, obj::SparseProjectiveHamiltonian)
	println(io, "$(typeof(obj)): site = $(obj.si), total channels = $(length(_countIntr(obj)))")
	_showDinfo(io, obj)
end

function _showDinfo(io::IO, obj::SparseProjectiveHamiltonian{N}) where {N}

	idx = findfirst(i -> !isnothing(obj.El[i]), 1:length(obj.El))
	D, DD = dim(obj.El[idx], numind(obj.El[idx]))
	println(io, "State[L]: $(domain(obj.El[idx])[end]), dim = $(D) -> $(DD)")
	D, DD = dim(obj.Er[1], 1)
	println(io, "State[R]: $(domain(obj.Er[idx])[end]), dim = $(D) -> $(DD)")
	for i in 1:N
		DL, DDL = dim(obj.H[i], 1)
		DR, DDR = dim(obj.H[i], 2)
		println(io, "Ham[site = $(obj.si[1] + i - 1)]: $(sum(DL)) × $(sum(DR)) -> $(sum(DDL)) × $(sum(DDR)) ($DL × $DR -> $DDL × $DDR)")
	end
	return nothing
end


function _countIntr(obj::SparseProjectiveHamiltonian{2})
	# count the valid interactions
	validIdx = Vector{NTuple{3, Int64}}(undef, 0)
	lscost = Int64[]
	for i in 1:length(obj.El), j in 1:size(obj.H[1], 2), k in 1:length(obj.Er)
		isnothing(obj.El[i]) && continue
		isnothing(obj.H[1][i, j]) && continue
		isnothing(obj.H[2][j, k]) && continue
		isnothing(obj.Er[k]) && continue
		push!(validIdx, (i, j, k))

		cost = numind(obj.El[i]) + numind(obj.Er[k])
		if !isa(obj.H[1][i, j], IdentityOperator)
			cost += numind(obj.H[1][i, j]) - 2
		end
		if !isa(obj.H[2][j, k], IdentityOperator)
			cost += numind(obj.H[2][j, k]) - 2
		end
		push!(lscost, cost)
	end
	# sort
	perms = sortperm(lscost; rev = true)
	return validIdx[perms]
end

function _countIntr(obj::SparseProjectiveHamiltonian{1})
	validIdx = Vector{NTuple{2, Int64}}(undef, 0)
	lscost = Int64[]
	for i in 1:length(obj.El), j in 1:length(obj.Er)
		isnothing(obj.El[i]) && continue
		isnothing(obj.H[1][i, j]) && continue
		isnothing(obj.Er[j]) && continue
		push!(validIdx, (i, j))

		cost = numind(obj.El[i]) + numind(obj.Er[j])
		if !isa(obj.H[1][i, j], IdentityOperator)
			cost += numind(obj.H[1][i, j]) - 2
		end
		push!(lscost, cost)
	end
	# sort
	perms = sortperm(lscost; rev = true)
	return validIdx[perms]
end

function _countIntr(obj::SparseProjectiveHamiltonian{0})
	validIdx = Tuple{Int64}[]
	lscost = Int64[]
	for i in 1:length(obj.El)
		isnothing(obj.El[i]) && continue
		isnothing(obj.Er[i]) && continue
		push!(validIdx, (i,))
		cost = numind(obj.El[i]) + numind(obj.Er[i])
		push!(lscost, cost)
	end
	# sort
	perms = sortperm(lscost; rev = true)
	return validIdx[perms]
end

function _countIntr(El::SparseLeftTensor, Er::SparseRightTensor, H::NTuple{2, SparseMPOTensor})
	# count the valid interactions
	validIdx = NTuple{3, Int64}[]
	lscost = Int64[]
	for i in 1:length(El), j in 1:size(H[1], 2), k in 1:length(Er)
		isnothing(El[i]) && continue
		isnothing(H[1][i, j]) && continue
		isnothing(H[2][j, k]) && continue
		isnothing(Er[k]) && continue
		push!(validIdx, (i, j, k))

		cost = numind(El[i]) + numind(Er[k])
		if !isa(H[1][i, j], IdentityOperator)
			cost += numind(H[1][i, j]) - 1
		end
		if !isa(H[2][j, k], IdentityOperator)
			cost += numind(H[2][j, k]) - 1
		end
		push!(lscost, cost)
	end
	# sort
	perms = sortperm(lscost; rev = true)
	return validIdx[perms]
end

function _countIntr(El::SparseLeftTensor, Er::SparseRightTensor, H::NTuple{1, SparseMPOTensor})
	# count the valid interactions
	validIdx = NTuple{2, Int64}[]
	lscost = Int64[]
	for i in 1:length(El), j in 1:length(Er)
		isnothing(El[i]) && continue
		isnothing(H[1][i, j]) && continue
		isnothing(Er[j]) && continue
		push!(validIdx, (i, j))

		cost = numind(El[i]) + numind(Er[j])
		if !isa(H[1][i, j], IdentityOperator)
			cost += numind(H[1][i, j]) - 1
		end
		push!(lscost, cost)
	end
	# sort
	perms = sortperm(lscost; rev = true)
	return validIdx[perms]
end

function _countIntr(El::SparseLeftTensor, Er::SparseRightTensor, ::NTuple{0, SparseMPOTensor})
	validIdx = Tuple{Int64}[]
	lscost = Int64[]
	for i in 1:length(El)
		isnothing(El[i]) && continue
		isnothing(Er[i]) && continue
		push!(validIdx, (i,))
		cost = numind(El[i]) + numind(Er[i])
		push!(lscost, cost)
	end
	# sort
	perms = sortperm(lscost; rev = true)
	return validIdx[perms]
end