"""
	 abstract type AbstractProjectiveHamiltonian
Abstract type of all projective Hamiltonian.
"""
abstract type AbstractProjectiveHamiltonian end

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

mutable struct CompositeProjectiveHamiltonian{L, N} <: AbstractProjectiveHamiltonian
	PH::Vector{SimpleProjectiveHamiltonian{L}}
	CI::NTuple{N, Channel}
	CO::NTuple{N, Channel}
	tasks::NTuple{N, Task}
	E₀::Float64
	function CompositeProjectiveHamiltonian(
		El::SparseLeftTensor,
		Er::SparseRightTensor,
		H::NTuple{L, SparseMPOTensor},
		E₀::Float64 = 0.0) where L

		validIdx = _countIntr(El, Er, H)
		if get_num_workers() == 1 # multithreading
			N = min(get_num_threads_action(), length(validIdx))
		else
			@assert false "TODO: multiprocessing!"
		end

		lsPH = map(validIdx) do idx
			Hi = map(1:L) do i
				H[i][idx[i], idx[i+1]]
			end
			SimpleProjectiveHamiltonian(El[idx[1]], Er[idx[end]], Hi...)
		end

		# static distributing
		lsids = map(Tuple(1:N)) do i
			Int64[]
		end
		i = idx_next = 1
		d = 1
		while i ≤ length(validIdx)
			push!(lsids[idx_next], i)
			# snake-like
			if i % N == 0
				d *= -1
			else
				idx_next += d
			end
			i += 1
		end

		# input channels 
		CI = map(Tuple(1:N)) do _
			Channel(0)
		end
		# output channels
		CO = map(Tuple(1:N)) do _
			Channel(0)
		end

		# tasks
		tasks = map(lsids, CI, CO) do ids, c_in, c_out
			t = Threads.@spawn begin
				# # task local PH 
				# local lsPH = map(validIdx[ids]) do idx
				# 	Hi = map(1:L) do i
				# 		H[i][idx[i], idx[i+1]]
				# 	end
				# 	SimpleProjectiveHamiltonian(El[idx[1]], Er[idx[end]], Hi...)
				# end

				while isopen(c_in)
					# keep waiting state tensor
					x = take!(c_in)
					if isnothing(x)
						# signal to kill the task
						# for PH in lsPH
						# 	finalize(PH)
						# end
						break
					end

					try
						# apply the action one by one 
						s = zero(x)
						for PH in lsPH[ids]
							add!(s, action(x, PH))
						end
						# put the result to c_out 
						put!(c_out, s)
					catch e
						# for PH in lsPH
						# 	finalize(PH)
						# end
						# put the error to c_out
						put!(c_out, e)
						rethrow(e)
					end
				end
			end
               errormonitor(t)
		end

		obj = new{L, N}(lsPH, CI, CO, tasks, E₀)

		# finalizer 
		finalizer(obj) do o
			# sent nothing to kill the task 
			# note the channel will be closed automatically after the task finish
			for c in o.CI
				Threads.@spawn put!(c, nothing)
			end
			for c in o.CO
				Threads.@spawn close(c)
			end
			for PH in o.PH
				finalize(PH)
			end
			wait.(o.tasks)
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
	D, DD = dim(obj.El[idx], rank(obj.El[idx]))
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

		cost = rank(obj.El[i]) + rank(obj.Er[k])
		if !isa(obj.H[1][i, j], IdentityOperator)
			cost += rank(obj.H[1][i, j]) - 2
		end
		if !isa(obj.H[2][j, k], IdentityOperator)
			cost += rank(obj.H[2][j, k]) - 2
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

		cost = rank(obj.El[i]) + rank(obj.Er[j])
		if !isa(obj.H[1][i, j], IdentityOperator)
			cost += rank(obj.H[1][i, j]) - 2
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
		cost = rank(obj.El[i]) + rank(obj.Er[i])
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

		cost = rank(El[i]) + rank(Er[k])
		if !isa(H[1][i, j], IdentityOperator)
			cost += rank(H[1][i, j]) - 1
		end
		if !isa(H[2][j, k], IdentityOperator)
			cost += rank(H[2][j, k]) - 1
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

		cost = rank(El[i]) + rank(Er[j])
		if !isa(H[1][i, j], IdentityOperator)
			cost += rank(H[1][i, j]) - 1
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
		cost = rank(El[i]) + rank(Er[i])
		push!(lscost, cost)
	end
	# sort
	perms = sortperm(lscost; rev = true)
	return validIdx[perms]
end