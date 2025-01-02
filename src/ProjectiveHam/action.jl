function action!(x::AbstractMPSTensor, PH::CompositeProjectiveHamiltonian)

	if get_num_workers() > 1 # multi-processing
          @assert false "TODO: multi-processing action!"
     elseif get_num_threads_action() == 1 # serial 
          
          x_cp = deepcopy(x)
          rmul!(x, -PH.E₀)
          for PH_i in PH.PH
               Hx_i = action(x_cp, PH_i)
               add!(x, Hx_i)
          end
               
	else # multi-threading
		x_cp = deepcopy(x)
		# - E₀x 
		rmul!(x, -PH.E₀)

		Lock = ReentrantLock()
		Threads.@threads :greedy for PH_i in PH.PH
			Hx_i = action(x_cp, PH_i)

			# reduce 
			lock(Lock)
			try
				add!(x, Hx_i)
			catch e
				rethrow(e)
			finally
				unlock(Lock)
			end
		end
	end
	return x
end
action(x::AbstractMPSTensor, PH::CompositeProjectiveHamiltonian) = action!(deepcopy(x), PH)


function action!(x::AbstractMPSTensor, PH::SimpleProjectiveHamiltonian)
	return _action!(x, PH.El, PH.H..., PH.Er, PH.cache)
end
action(x::AbstractMPSTensor, PH::SimpleProjectiveHamiltonian) = action!(deepcopy(x), PH)

# ======================== 2-site MPS ========================
# TODO: change to _action!(Hx, x, ..., cache, α, β) -> write α H * x + β Hx to Hx
function _action!(x::CompositeMPSTensor{2, T}, El::LocalLeftTensor{3}, Hl::LocalOperator{2, 1}, Hr::IdentityOperator, Er::LocalRightTensor{2}, cache::Vector{<:AbstractTensorMap}) where T <: NTuple{2, MPSTensor{3}}

	if isempty(cache)
		# [1] permute El
		push!(cache, permute(El.A, ((1, 2), (3,))))
		# [2] permute x
		push!(cache, permute(x.A, ((1,), (2, 3, 4))))
		# [3] El * x 
		push!(cache, cache[1] * cache[2])
		# [4] permute El*x
		push!(cache, permute(cache[3], ((1, 4, 5), (2, 3))))
		# [5] permute Hl 
		push!(cache, permute(Hl.A, ((1, 3), (2,))))
		# [6] contract Hl
		push!(cache, cache[4] * cache[5])
		# [7] permute El*x*Hl
		push!(cache, permute(cache[6], ((1, 4, 2), (3,))))
		# [8] contract Er 
		push!(cache, Hl.strength * Hr.strength * cache[7] * Er.A)
	else
		# permute x 
		permute!(cache[2], x.A, ((1,), (2, 3, 4)))
		# El * x  
		mul!(cache[3], cache[1], cache[2])
		# permute El*x
		permute!(cache[4], cache[3], ((1, 4, 5), (2, 3)))
		# contract Hl
		mul!(cache[6], cache[4], cache[5])
		# permute El*x*Hl
		permute!(cache[7], cache[6], ((1, 4, 2), (3,)))
		# contract Er 
		mul!(cache[8], cache[7], Er.A, Hl.strength * Hr.strength, 0.0)

	end

	permute!(x.A, cache[8], ((1, 2), (3, 4)))
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{3},
	Hl::LocalOperator{1, 1},
	Hr::LocalOperator{2, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap}) where T <: NTuple{2, MPSTensor{3}}

	# x, Er, Hl, Hr, El
	if isempty(cache)
		# [1] permute x 
		push!(cache, permute(x.A, ((1, 2, 3), (4,))))
		# [2] x * Er
		push!(cache, cache[1] * Er.A)
		# [3] permute x*Er
		push!(cache, permute(cache[2], ((2,), (1, 3, 4))))
		# [4] Hl*x*Er
		push!(cache, Hl.A * cache[3])
		# [5] permute Hl*x*Er
		push!(cache, permute(cache[4], ((3,), (2, 1, 4))))
		# [6] Hl*x*Er*Hr
		push!(cache, Hr.A * cache[5])
		# [7] permute Hl*x*Er*Hr
		push!(cache, permute(cache[6], ((1, 3), (4, 2, 5))))
		# [8] El *Hl*x*Er*Hr
		push!(cache, Hl.strength * Hr.strength * El.A * cache[7])
	else
		# permute x 
		permute!(cache[1], x.A, ((1, 2, 3), (4,)))
		# x * Er
		mul!(cache[2], cache[1], Er.A)
		# permute x*Er
		permute!(cache[3], cache[2], ((2,), (1, 3, 4)))
		# Hl*x*Er
		mul!(cache[4], Hl.A, cache[3])
		# permute Hl*x*Er
		permute!(cache[5], cache[4], ((3,), (2, 1, 4)))
		# Hl*x*Er*Hr
		mul!(cache[6], Hr.A, cache[5])
		# permute Hl*x*Er*Hr
		permute!(cache[7], cache[6], ((1, 3), (4, 2, 5)))
		# El *Hl*x*Er*Hr
		mul!(cache[8], El.A, cache[7], Hl.strength * Hr.strength, 0.0)
	end

	permute!(x.A, cache[8], ((1, 2), (3, 4)))
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{3},
	Hl::LocalOperator{1, 1},
	Hr::LocalOperator{1, 1},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap}) where T <: NTuple{2, MPSTensor{3}}

	# x, Hl, Hr, El, Er 
	if isempty(cache)
		# [1] permute x
		push!(cache, permute(x.A, ((2,), (1, 3, 4))))
		# [2] Hl*x
		push!(cache, Hl.A * cache[1])
		# [3] permute Hl*x
		push!(cache, permute(cache[2], ((3,), (2, 1, 4))))
		# [4] Hl*x*Hr
		push!(cache, Hr.A * cache[3])
		# [5] permute Hl*x*Hr
		push!(cache, permute(cache[4], ((2,), (3, 1, 4))))
		# [6] permute El
		push!(cache, permute(El.A, ((1, 2), (3,))))
		# [7] El*Hl*x*Hr
		push!(cache, cache[6] * cache[5])
		# [8] permute El*Hl*x*Hr
		push!(cache, permute(cache[7], ((1, 3, 4), (5, 2))))
		# [9] El*Hl*x*Hr*Er
		push!(cache, Hl.strength * Hr.strength * cache[8] * Er.A)
	else
		# permute x
		permute!(cache[1], x.A, ((2,), (1, 3, 4)))
		# Hl*x
		mul!(cache[2], Hl.A, cache[1])
		# permute Hl*x
		permute!(cache[3], cache[2], ((3,), (2, 1, 4)))
		# Hl*x*Hr
		mul!(cache[4], Hr.A, cache[3])
		# permute Hl*x*Hr
		permute!(cache[5], cache[4], ((2,), (3, 1, 4)))
		# El*Hl*x*Hr
		mul!(cache[7], cache[6], cache[5])
		# permute El*Hl*x*Hr
		permute!(cache[8], cache[7], ((1, 3, 4), (5, 2)))
		# El*Hl*x*Hr*Er
		mul!(cache[9], cache[8], Er.A, Hl.strength * Hr.strength, 0.0)

	end

	permute!(x.A, cache[9], ((1, 2), (3, 4)))
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::LocalOperator{1, 2},
	Hr::LocalOperator{1, 1},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap}) where T <: NTuple{2, MPSTensor{3}}

	# El, x, Hr, Hl, Er
	if isempty(cache)
		# [1] permute x 
		push!(cache, permute(x.A, ((1,), (2, 3, 4))))
		# [2] El * x
		push!(cache, El.A * cache[1])
		# [3] permute El*x
		push!(cache, permute(cache[2], ((3,), (1, 2, 4))))
		# [4] Hr * El * x
		push!(cache, Hr.A * cache[3])
		# [5] permute Hr * El * x
		push!(cache, permute(cache[4], ((3,), (2, 1, 4))))
		# [6] permute Hl 
		push!(cache, permute(Hl.A, ((1, 3), (2,))))
		# [7] Hl * Hr * El * x
		push!(cache, cache[6] * cache[5])
		# [8] permute Hl * Hr * El * x
		push!(cache, permute(cache[7], ((3, 1, 4), (5, 2))))
		# [9] Hl * Hr * El * x * Er
		push!(cache, Hl.strength * Hr.strength * cache[8] * Er.A)
	else
		# permute x 
		permute!(cache[1], x.A, ((1,), (2, 3, 4)))
		# El * x
		mul!(cache[2], El.A, cache[1])
		# permute El*x
		permute!(cache[3], cache[2], ((3,), (1, 2, 4)))
		# Hr * El * x
		mul!(cache[4], Hr.A, cache[3])
		# permute Hr * El * x
		permute!(cache[5], cache[4], ((3,), (2, 1, 4)))
		# Hl * Hr * El * x
		mul!(cache[7], cache[6], cache[5])
		# permute Hl * Hr * El * x
		permute!(cache[8], cache[7], ((3, 1, 4), (5, 2)))
		# Hl * Hr * El * x * Er
		mul!(cache[9], cache[8], Er.A, Hl.strength * Hr.strength, 0.0)

	end

	permute!(x.A, cache[9], ((1, 2), (3, 4)))
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::IdentityOperator,
	Hr::LocalOperator{1, 2},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap}) where T <: NTuple{2, MPSTensor{3}}

	# El, x, Hr, Er
	if isempty(cache)
		# [1] permute x 
		push!(cache, permute(x.A, ((1,), (2, 3, 4))))
		# [2] El * x
		push!(cache, El.A * cache[1])
		# [3] permute El*x
		push!(cache, permute(cache[2], ((3,), (1, 2, 4))))
		# [4] permute Hr
		push!(cache, permute(Hr.A, ((1, 3), (2,))))
		# [5] Hr * El * x
		push!(cache, cache[4] * cache[3])
		# [6] permute Hr * El * x
		push!(cache, permute(cache[5], ((3, 4, 1), (5, 2))))
		# [7] Hr * El * x * Er
		push!(cache, Hl.strength * Hr.strength * cache[6] * Er.A)
	else
		# permute x 
		permute!(cache[1], x.A, ((1,), (2, 3, 4)))
		# El * x
		mul!(cache[2], El.A, cache[1])
		# permute El*x
		permute!(cache[3], cache[2], ((3,), (1, 2, 4)))
		# permute Hr
		permute!(cache[4], Hr.A, ((1, 3), (2,)))
		# Hr * El * x
		mul!(cache[5], cache[4], cache[3])
		# permute Hr * El * x
		permute!(cache[6], cache[5], ((3, 4, 1), (5, 2)))
		# Hr * El * x * Er
		mul!(cache[7], cache[6], Er.A, Hl.strength * Hr.strength, 0.0)
	end

	permute!(x.A, cache[7], ((1, 2), (3, 4)))
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::IdentityOperator,
	Hr::IdentityOperator,
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap}) where T <: NTuple{2, MPSTensor{3}}

	# El, x, Er 
	if isempty(cache)
		# [1] permute x 
		push!(cache, permute(x.A, ((1,), (2, 3, 4))))
		# [2] El * x
		push!(cache, El.A * cache[1])
		# [3] permute El*x
		push!(cache, permute(cache[2], ((1, 2, 3), (4,))))
		# [4] El * x * Er
		push!(cache, Hl.strength * Hr.strength * cache[3] * Er.A)
	else
		# permute x 
		permute!(cache[1], x.A, ((1,), (2, 3, 4)))
		# El * x
		mul!(cache[2], El.A, cache[1])
		# permute El*x
		permute!(cache[3], cache[2], ((1, 2, 3), (4,)))
		# El * x * Er
		mul!(cache[4], cache[3], Er.A, Hl.strength * Hr.strength, 0.0)
	end

	permute!(x.A, cache[4], ((1, 2), (3, 4)))
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::LocalOperator{1, 1},
	Hr::IdentityOperator,
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap}) where T <: NTuple{2, MPSTensor{3}}

	# El, x, Hl, Er
	if isempty(cache)
		# [1] permute x 
		push!(cache, permute(x.A, ((1,), (2, 3, 4))))
		# [2] El * x
		push!(cache, El.A * cache[1])
		# [3] permute El*x
		push!(cache, permute(cache[2], ((2,), (1, 3, 4))))
		# [4] Hl * El * x
		push!(cache, Hl.A * cache[3])
		# [5] permute Hl * El * x
		push!(cache, permute(cache[4], ((2, 1, 3), (4,))))
		# [6] Hl * El * x * Er
		push!(cache, Hl.strength * Hr.strength * cache[5] * Er.A)
	else
		# permute x 
		permute!(cache[1], x.A, ((1,), (2, 3, 4)))
		# El * x
		mul!(cache[2], El.A, cache[1])
		# permute El*x
		permute!(cache[3], cache[2], ((2,), (1, 3, 4)))
		# Hl * El * x
		mul!(cache[4], Hl.A, cache[3])
		# permute Hl * El * x
		permute!(cache[5], cache[4], ((2, 1, 3), (4,)))
		# Hl * El * x * Er
		mul!(cache[6], cache[5], Er.A, Hl.strength * Hr.strength, 0.0)
	end

	permute!(x.A, cache[6], ((1, 2), (3, 4)))
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::IdentityOperator,
	Hr::LocalOperator{1, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap}) where T <: NTuple{2, MPSTensor{3}}

	# El, x, Hr, Er
	if isempty(cache)
		# [1] permute x 
		push!(cache, permute(x.A, ((1,), (2, 3, 4))))
		# [2] El * x
		push!(cache, El.A * cache[1])
		# [3] permute El*x
		push!(cache, permute(cache[2], ((3,), (1, 2, 4))))
		# [4] Hr * El * x
		push!(cache, Hr.A * cache[3])
		# [5] permute Hr * El * x
		push!(cache, permute(cache[4], ((2, 3, 1), (4,))))
		# [6] Hr * El * x * Er
		push!(cache, Hl.strength * Hr.strength * cache[5] * Er.A)
	else
		# permute x 
		permute!(cache[1], x.A, ((1,), (2, 3, 4)))
		# El * x
		mul!(cache[2], El.A, cache[1])
		# permute El*x
		permute!(cache[3], cache[2], ((3,), (1, 2, 4)))
		# Hr * El * x
		mul!(cache[4], Hr.A, cache[3])
		# permute Hr * El * x
		permute!(cache[5], cache[4], ((2, 3, 1), (4,)))
		# Hr * El * x * Er
		mul!(cache[6], cache[5], Er.A, Hl.strength * Hr.strength, 0.0)
	end

	permute!(x.A, cache[6], ((1, 2), (3, 4)))
	return x

end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::LocalOperator{1, 2},
	Hr::LocalOperator{2, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap}) where T <: NTuple{2, MPSTensor{3}}

	# El x, Hl, Hr, Er
	if isempty(cache)
		# [1] permute x 
		push!(cache, permute(x.A, ((1,), (2, 3, 4))))
		# [2] El * x
		push!(cache, El.A * cache[1])
		# [3] permute El * x
		push!(cache, permute(cache[2], ((2,), (1, 3, 4))))
		# [4] permute Hl 
		push!(cache, permute(Hl.A, ((1, 3), (2,))))
		# [5] Hl * El * x
		push!(cache, cache[4] * cache[3])
		# [6] permute Hl * El * x
		push!(cache, permute(cache[5], ((3, 1, 5), (2, 4))))
		# [7] permute Hr
		push!(cache, permute(Hr.A, ((1, 3), (2,))))
		# [8] Hl * El * x * Hr
		push!(cache, cache[6] * cache[7])
		# [9] permute Hl * El * x * Hr
		push!(cache, permute(cache[8], ((1, 2, 4), (3,))))
		# [10] Hl * El * x * Hr * Er
		push!(cache, Hl.strength * Hr.strength * cache[9] * Er.A)
	else
		# permute x 
		permute!(cache[1], x.A, ((1,), (2, 3, 4)))
		# El * x
		mul!(cache[2], El.A, cache[1])
		# permute El * x
		permute!(cache[3], cache[2], ((2,), (1, 3, 4)))
		# Hl * El * x
		mul!(cache[5], cache[4], cache[3])
		# permute Hl * El * x
		permute!(cache[6], cache[5], ((3, 1, 5), (2, 4)))
		# Hl * El * x * Hr
		mul!(cache[8], cache[6], cache[7])
		# permute Hl * El * x * Hr
		permute!(cache[9], cache[8], ((1, 2, 4), (3,)))
		# Hl * El * x * Hr * Er
		mul!(cache[10], cache[9], Er.A, Hl.strength * Hr.strength, 0.0)
	end

	permute!(x.A, cache[10], ((1, 2), (3, 4)))
	return x
end

# ======================== 1-site MPO ========================
function _action!(x::MPSTensor{4}, El::LocalLeftTensor{2}, H::LocalOperator{1, 2}, Er::LocalRightTensor{3}, cache::Vector{<:AbstractTensorMap})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# El, x, H, Er 
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4))))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1]))
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4))))
		# [4] permute H 
		push!(cache, _permute_malloc(TT, H.A, ((1, 3), (2,))))
		# [5] H * El * x
		push!(cache, _mul_malloc(TT, cache[4], cache[3]))
		# [6] permute H * El * x
		push!(cache, _permute_malloc(TT, cache[5], ((3, 1, 4), (5, 2))))
		# [7] H * El * x * Er
		push!(cache, _mul_malloc(TT, cache[6], Er.A, H.strength))
	else
		# permute x
		permute!(cache[1], x.A, ((1,), (2, 3, 4)))
		# El * x
		mul!(cache[2], El.A, cache[1])
		# permute El * x
		permute!(cache[3], cache[2], ((2,), (1, 3, 4)))
		# H * El * x
		mul!(cache[5], cache[4], cache[3])
		# permute H * El * x
		permute!(cache[6], cache[5], ((3, 1, 4), (5, 2)))
		# H * El * x * Er
		mul!(cache[7], cache[6], Er.A, H.strength, 0.0)
	end

	permute!(x.A, cache[7], ((1, 2), (3, 4)))
	return x
end

function _action!(x::MPSTensor{4}, El::LocalLeftTensor{2}, H::IdentityOperator, Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap})

	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	# El, x, Er 
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4))))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1]))
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((1, 2, 3), (4,))))
		# [4] El * x * Er
		push!(cache, _mul_malloc(TT, cache[3], Er.A, H.strength))
	else
		# permute x
		permute!(cache[1], x.A, ((1,), (2, 3, 4)))
		# El * x
		mul!(cache[2], El.A, cache[1])
		# permute El * x
		permute!(cache[3], cache[2], ((1, 2, 3), (4,)))
		# El * x * Er
		mul!(cache[4], cache[3], Er.A, H.strength, 0.0)
	end

	permute!(x.A, cache[4], ((1, 2), (3, 4)))
	return x
end

function _action!(x::MPSTensor{4}, El::LocalLeftTensor{2}, H::LocalOperator{1, 1}, Er::LocalRightTensor{2}, cache::Vector{<:AbstractTensorMap})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# H, x, El, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((2,), (1, 3, 4))))
		# [2] H * x
		push!(cache, _mul_malloc(TT, H.A, cache[1]))
		# [3] permute H * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4))))
		# [4] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[3]))
		# [5] permute El * x
		push!(cache, _permute_malloc(TT, cache[4], ((1, 2, 3), (4,))))
		# [6] El * x * Er
		push!(cache, _mul_malloc(TT, cache[5], Er.A, H.strength))
	else
		# permute x
		permute!(cache[1], x.A, ((2,), (1, 3, 4)))
		# H * x
		mul!(cache[2], H.A, cache[1])
		# permute H * x
		permute!(cache[3], cache[2], ((2,), (1, 3, 4)))
		# El * x
		mul!(cache[4], El.A, cache[3])
		# permute El * x
		permute!(cache[5], cache[4], ((1, 2, 3), (4,)))
		# El * x * Er
		mul!(cache[6], cache[5], Er.A, H.strength, 0.0)
	end

	permute!(x.A, cache[6], ((1, 2), (3, 4)))
	return x
end

function _action!(x::MPSTensor{4}, El::LocalLeftTensor{3}, H::LocalOperator{2, 1}, Er::LocalRightTensor{2}, cache::Vector{<:AbstractTensorMap})

	# x, Er, H, El
	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1, 2, 3), (4,))))
		# [2] x * Er
		push!(cache, _mul_malloc(TT, cache[1], Er.A))
		# [3] permute x * Er
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4))))
		# [4] H * x * Er
		push!(cache, _mul_malloc(TT, H.A, cache[3]))
		# [5] permute H * x * Er
		push!(cache, _permute_malloc(TT, cache[4], ((1, 3), (2, 4, 5))))
		# [6] El * x * Er
		push!(cache, _mul_malloc(TT, El.A, cache[5], H.strength))
	else
		# permute x
		permute!(cache[1], x.A, ((1, 2, 3), (4,)))
		# x * Er
		mul!(cache[2], cache[1], Er.A)
		# permute x * Er
		permute!(cache[3], cache[2], ((2,), (1, 3, 4)))
		# H * x * Er
		mul!(cache[4], H.A, cache[3])
		# permute H * x * Er
		permute!(cache[5], cache[4], ((1, 3), (2, 4, 5)))
		# El * x * Er
		mul!(cache[6], El.A, cache[5], H.strength, 0.0)
	end

	permute!(x.A, cache[6], ((1, 2), (3, 4)))
	return x
end

function _action!(x::MPSTensor{4}, El::LocalLeftTensor{3}, H::LocalOperator{1, 1}, Er::LocalRightTensor{3}, cache::Vector{<:AbstractTensorMap})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# H, x, El, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((2,), (1, 3, 4))))
		# [2] H * x
		push!(cache, _mul_malloc(TT, H.A, cache[1]))
		# [3] permute H * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4))))
		# [4] permute El 
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,))))
		# [5] El * H * x
		push!(cache, _mul_malloc(TT, cache[4], cache[3]))
		# [6] permute El * H * x
		push!(cache, _permute_malloc(TT, cache[5], ((1, 3, 4), (5, 2))))
		# [7] El * H * x * Er
		push!(cache, _mul_malloc(TT, cache[6], Er.A, H.strength))
	else
		# permute x
		permute!(cache[1], x.A, ((2,), (1, 3, 4)))
		# H * x
		mul!(cache[2], H.A, cache[1])
		# permute H * x
		permute!(cache[3], cache[2], ((2,), (1, 3, 4)))
		# El * H * x
		mul!(cache[5], cache[4], cache[3])
		# permute El * H * x
		permute!(cache[6], cache[5], ((1, 3, 4), (5, 2)))
		# El * H * x * Er
		mul!(cache[7], cache[6], Er.A, H.strength, 0.0)
	end

	permute!(x.A, cache[7], ((1, 2), (3, 4)))
	return x
end

function _action!(x::MPSTensor{4}, El::LocalLeftTensor{3}, H::IdentityOperator, Er::LocalRightTensor{3}, cache::Vector{<:AbstractTensorMap})

	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	# El, x, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4))))
		# [2] permute El
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,))))
		# [3] El * x
		push!(cache, _mul_malloc(TT, cache[2], cache[1]))
		# [4] permute El * x
		push!(cache, _permute_malloc(TT, cache[3], ((1, 3, 4), (5, 2))))
		# [5] El * x * Er
		push!(cache, _mul_malloc(TT, cache[4], Er.A, H.strength))
	else
		# permute x
		permute!(cache[1], x.A, ((1,), (2, 3, 4)))
		# El * x
		mul!(cache[3], cache[2], cache[1])
		# permute El * x
		permute!(cache[4], cache[3], ((1, 3, 4), (5, 2)))
		# El * x * Er
		mul!(cache[5], cache[4], Er.A, H.strength, 0.0)
	end

	permute!(x.A, cache[5], ((1, 2), (3, 4)))
	return x
end

# ======================== bond ========================
function _action!(x::MPSTensor{2}, El::LocalLeftTensor{3}, Er::LocalRightTensor{3}, cache::Vector{<:AbstractTensorMap})

	# El, x, Er
	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	if isempty(cache)
		# [1] permute El
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,))))
		# [2] El * x 
		push!(cache, cache[1] * x.A)
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((1,), (3, 2))))
	else
		# El * x 
		mul!(cache[2], cache[1], x.A)
		# permute El * x
		permute!(cache[3], cache[2], ((1,), (3, 2)))
	end

	# El * x * Er
	mul!(x.A, cache[3], Er.A)
	return x
end
function _action!(x::MPSTensor{2}, El::LocalLeftTensor{2}, Er::LocalRightTensor{2}, cache::Vector{<:AbstractTensorMap})

	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	# El, x, Er
	if isempty(cache)
		# [1] El * x
		push!(cache, El.A * x.A)
	else
		# El * x 
		mul!(cache[1], El.A, x.A)
	end

	# El * x * Er
	mul!(x.A, cache[1], Er.A)
	return x
end

function _permute_malloc(
	T::Type{<:Union{Float64, ComplexF64}},
	A::AbstractTensorMap,
	pA::Index2Tuple)
	# alloc a new tensor with manual allocator
	t = tensoralloc_add(T, A, pA, false, Val(true), ManualAllocator())
	permute!(t, A, pA)
	return t
end

function _mul_malloc(
	T::Type{<:Union{Float64, ComplexF64}},
	A::AbstractTensorMap,
	B::AbstractTensorMap,
	α::Number = 1.0)
	# α * A * B with manual allocator
	s = HomSpace(codomain(A), domain(B))
	d = fusionblockstructure(s).totaldim
	data = tensoralloc(Vector{T}, d, Val(true), ManualAllocator())
	t = TensorMap{T}(data, s)
	rmul!(t, 0.0)
	mul!(t, A, B, α)
	return t
end

tensorfree!(x::Array{<:Union{Float64, ComplexF64}}, ::ManualAllocator) = nothing
