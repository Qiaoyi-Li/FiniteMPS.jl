function action!(x::AbstractMPSTensor, PH::CompositeProjectiveHamiltonian, TO::Union{TimerOutput, Nothing} = nothing)

	if get_num_workers() > 1 # multi-processing
		@assert false "TODO: multi-processing action!"
	elseif get_num_threads_action() == 1 # serial 

		x_cp = deepcopy(x)
		rmul!(x, -PH.E₀)
		for PH_i in PH.PH
			Hx_i = action(x_cp, PH_i, TO)
			add!(x, Hx_i)
		end

	else # multi-threading
		x_cp = deepcopy(x)
		# - E₀x 
		rmul!(x, -PH.E₀)

		TO_tot = TimerOutput()
		TO_reduce = TimerOutput()
		@timeit TO_tot "action" begin
			Lock = ReentrantLock()
			Threads.@threads :greedy for PH_i in PH.PH

				if isnothing(TO)
					Hx_i = action(x_cp, PH_i, nothing)
				else
					str = _action_str(PH_i)
					TO_i = TimerOutput()
					@timeit TO_i str Hx_i = action(x_cp, PH_i, TO_i)
				end

				# reduce 
				lock(Lock)
				try
					add!(x, Hx_i)
					if !isnothing(TO)
						merge!(TO_reduce, TO_i)
					end
				catch e
					rethrow(e)
				finally
					unlock(Lock)
				end
			end
		end
		if !isnothing(TO)
			merge!(TO_tot, TO_reduce; tree_point = ["action"])
			merge!(TO, TO_tot; tree_point = [TO.prev_timer_label])
		end
	end
	return x
end
action(x::AbstractMPSTensor, PH::CompositeProjectiveHamiltonian, TO::Union{TimerOutput, Nothing} = nothing) = action!(deepcopy(x), PH, TO)


function action!(x::AbstractMPSTensor, PH::SimpleProjectiveHamiltonian, TO::Union{TimerOutput, Nothing} = nothing)
	return _action!(x, PH.El, PH.H..., PH.Er, PH.cache, TO)
end
action(x::AbstractMPSTensor, PH::SimpleProjectiveHamiltonian, TO::Union{TimerOutput, Nothing} = nothing) = action!(deepcopy(x), PH, TO)

# ======================== 2-site MPS ========================
# TODO: change to _action!(Hx, x, ..., cache, α, β) -> write α H * x + β Hx to Hx
function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{3},
	Hl::LocalOperator{2, 1},
	Hr::IdentityOperator,
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hl), eltype(Er))
	if isempty(cache)
		# [1] permute El
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,)), TO))
		# [2] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [3] El * x 
		push!(cache, _mul_malloc(TT, cache[1], cache[2], TO))
		# [4] permute El*x
		push!(cache, _permute_malloc(TT, cache[3], ((1, 4, 5), (2, 3)), TO))
		# [5] permute Hl 
		push!(cache, _permute_malloc(TT, Hl.A, ((1, 3), (2,)), TO))
		# [6] contract Hl
		push!(cache, _mul_malloc(TT, cache[4], cache[5], TO))
		# [7] permute El*x*Hl
		push!(cache, _permute_malloc(TT, cache[6], ((1, 4, 2), (3,)), TO))
		# [8] contract Er 
		push!(cache, _mul_malloc(TT, cache[7], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x 
		_permute_TO!(cache[2], x.A, ((1,), (2, 3, 4)), TO)
		# El * x  
		_mul_TO!(cache[3], cache[1], cache[2], TO)
		# permute El*x
		_permute_TO!(cache[4], cache[3], ((1, 4, 5), (2, 3)), TO)
		# contract Hl
		_mul_TO!(cache[6], cache[4], cache[5], TO)
		# permute El*x*Hl
		_permute_TO!(cache[7], cache[6], ((1, 4, 2), (3,)), TO)
		# contract Er 
		_mul_TO!(cache[8], cache[7], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[8], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{3},
	Hl::LocalOperator{1, 1},
	Hr::LocalOperator{2, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hl), eltype(Hr), eltype(Er))
	# x, Er, Hl, Hr, El
	if isempty(cache)
		# [1] permute x 
		push!(cache, _permute_malloc(TT, x.A, ((1, 2, 3), (4,)), TO))
		# [2] x * Er
		push!(cache, _mul_malloc(TT, cache[1], Er.A, TO))
		# [3] permute x*Er
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4)), TO))
		# [4] Hl*x*Er
		push!(cache, _mul_malloc(TT, Hl.A, cache[3], TO))
		# [5] permute Hl*x*Er
		push!(cache, _permute_malloc(TT, cache[4], ((3,), (2, 1, 4)), TO))
		# [6] Hl*x*Er*Hr
		push!(cache, _mul_malloc(TT, Hr.A, cache[5], TO))
		# [7] permute Hl*x*Er*Hr
		push!(cache, _permute_malloc(TT, cache[6], ((1, 3), (4, 2, 5)), TO))
		# [8] El *Hl*x*Er*Hr
		push!(cache, _mul_malloc(TT, El.A, cache[7], Hl.strength[] * Hr.strength[], TO))
	else
		# permute x 
		_permute_TO!(cache[1], x.A, ((1, 2, 3), (4,)), TO)
		# x * Er
		_mul_TO!(cache[2], cache[1], Er.A, TO)
		# permute x*Er
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3, 4)), TO)
		# Hl*x*Er
		_mul_TO!(cache[4], Hl.A, cache[3], TO)
		# permute Hl*x*Er
		_permute_TO!(cache[5], cache[4], ((3,), (2, 1, 4)), TO)
		# Hl*x*Er*Hr
		_mul_TO!(cache[6], Hr.A, cache[5], TO)
		# permute Hl*x*Er*Hr
		_permute_TO!(cache[7], cache[6], ((1, 3), (4, 2, 5)), TO)
		# El *Hl*x*Er*Hr
		_mul_TO!(cache[8], El.A, cache[7], Hl.strength[] * Hr.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[8], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{3},
	Hl::LocalOperator{1, 1},
	Hr::LocalOperator{1, 1},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hl), eltype(Hr), eltype(Er))
	# x, Hl, Hr, El, Er 
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((2,), (1, 3, 4)), TO))
		# [2] Hl*x
		push!(cache, _mul_malloc(TT, Hl.A, cache[1], TO))
		# [3] permute Hl*x
		push!(cache, _permute_malloc(TT, cache[2], ((3,), (2, 1, 4)), TO))
		# [4] Hl*x*Hr
		push!(cache, _mul_malloc(TT, Hr.A, cache[3], TO))
		# [5] permute Hl*x*Hr
		push!(cache, _permute_malloc(TT, cache[4], ((2,), (3, 1, 4)), TO))
		# [6] permute El
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,)), TO))
		# [7] El*Hl*x*Hr
		push!(cache, _mul_malloc(TT, cache[6], cache[5], TO))
		# [8] permute El*Hl*x*Hr
		push!(cache, _permute_malloc(TT, cache[7], ((1, 3, 4), (5, 2)), TO))
		# [9] El*Hl*x*Hr*Er
		push!(cache, _mul_malloc(TT, cache[8], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((2,), (1, 3, 4)), TO)
		# Hl*x
		_mul_TO!(cache[2], Hl.A, cache[1], TO)
		# permute Hl*x
		_permute_TO!(cache[3], cache[2], ((3,), (2, 1, 4)), TO)
		# Hl*x*Hr
		_mul_TO!(cache[4], Hr.A, cache[3], TO)
		# permute Hl*x*Hr
		_permute_TO!(cache[5], cache[4], ((2,), (3, 1, 4)), TO)
		# El*Hl*x*Hr
		_mul_TO!(cache[7], cache[6], cache[5], TO)
		# permute El*Hl*x*Hr
		_permute_TO!(cache[8], cache[7], ((1, 3, 4), (5, 2)), TO)
		# El*Hl*x*Hr*Er
		_mul_TO!(cache[9], cache[8], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)

	end

	_permute_TO!(x.A, cache[9], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::LocalOperator{1, 2},
	Hr::LocalOperator{1, 1},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hl), eltype(Hr), eltype(Er))
	# El, x, Hr, Hl, Er
	if isempty(cache)
		# [1] permute x 
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El*x
		push!(cache, _permute_malloc(TT, cache[2], ((3,), (1, 2, 4)), TO))
		# [4] Hr * El * x
		push!(cache, _mul_malloc(TT, Hr.A, cache[3], TO))
		# [5] permute Hr * El * x
		push!(cache, _permute_malloc(TT, cache[4], ((3,), (2, 1, 4)), TO))
		# [6] permute Hl 
		push!(cache, _permute_malloc(TT, Hl.A, ((1, 3), (2,)), TO))
		# [7] Hl * Hr * El * x
		push!(cache, _mul_malloc(TT, cache[6], cache[5], TO))
		# [8] permute Hl * Hr * El * x
		push!(cache, _permute_malloc(TT, cache[7], ((3, 1, 4), (5, 2)), TO))
		# [9] Hl * Hr * El * x * Er
		push!(cache, _mul_malloc(TT, cache[8], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x 
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El*x
		_permute_TO!(cache[3], cache[2], ((3,), (1, 2, 4)), TO)
		# Hr * El * x
		_mul_TO!(cache[4], Hr.A, cache[3], TO)
		# permute Hr * El * x
		_permute_TO!(cache[5], cache[4], ((3,), (2, 1, 4)), TO)
		# Hl * Hr * El * x
		_mul_TO!(cache[7], cache[6], cache[5], TO)
		# permute Hl * Hr * El * x
		_permute_TO!(cache[8], cache[7], ((3, 1, 4), (5, 2)), TO)
		# Hl * Hr * El * x * Er
		_mul_TO!(cache[9], cache[8], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)

	end

	_permute_TO!(x.A, cache[9], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::IdentityOperator,
	Hr::LocalOperator{1, 2},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hr), eltype(Er))
	# El, x, Hr, Er
	if isempty(cache)
		# [1] permute x 
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El*x
		push!(cache, _permute_malloc(TT, cache[2], ((3,), (1, 2, 4)), TO))
		# [4] permute Hr
		push!(cache, _permute_malloc(TT, Hr.A, ((1, 3), (2,)), TO))
		# [5] Hr * El * x
		push!(cache, _mul_malloc(TT, cache[4], cache[3], TO))
		# [6] permute Hr * El * x
		push!(cache, _permute_malloc(TT, cache[5], ((3, 4, 1), (5, 2)), TO))
		# [7] Hr * El * x * Er
		push!(cache, _mul_malloc(TT, cache[6], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x 
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El*x
		_permute_TO!(cache[3], cache[2], ((3,), (1, 2, 4)), TO)
		# permute Hr
		_permute_TO!(cache[4], Hr.A, ((1, 3), (2,)), TO)
		# Hr * El * x
		_mul_TO!(cache[5], cache[4], cache[3], TO)
		# permute Hr * El * x
		_permute_TO!(cache[6], cache[5], ((3, 4, 1), (5, 2)), TO)
		# Hr * El * x * Er
		_mul_TO!(cache[7], cache[6], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[7], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::IdentityOperator,
	Hr::IdentityOperator,
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	# El, x, Er 
	if isempty(cache)
		# [1] permute x 
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El*x
		push!(cache, _permute_malloc(TT, cache[2], ((1, 2, 3), (4,)), TO))
		# [4] El * x * Er
		push!(cache, _mul_malloc(TT, cache[3], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x 
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El*x
		_permute_TO!(cache[3], cache[2], ((1, 2, 3), (4,)), TO)
		# El * x * Er
		_mul_TO!(cache[4], cache[3], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[4], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::LocalOperator{1, 1},
	Hr::IdentityOperator,
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hl), eltype(Er))
	# El, x, Hl, Er
	if isempty(cache)
		# [1] permute x 
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El*x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4)), TO))
		# [4] Hl * El * x
		push!(cache, _mul_malloc(TT, Hl.A, cache[3], TO))
		# [5] permute Hl * El * x
		push!(cache, _permute_malloc(TT, cache[4], ((2, 1, 3), (4,)), TO))
		# [6] Hl * El * x * Er
		push!(cache, _mul_malloc(TT, cache[5], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x 
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El*x
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3, 4)), TO)
		# Hl * El * x
		_mul_TO!(cache[4], Hl.A, cache[3], TO)
		# permute Hl * El * x
		_permute_TO!(cache[5], cache[4], ((2, 1, 3), (4,)), TO)
		# Hl * El * x * Er
		_mul_TO!(cache[6], cache[5], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[6], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::IdentityOperator,
	Hr::LocalOperator{1, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hr), eltype(Er))
	# El, x, Hr, Er
	if isempty(cache)
		# [1] permute x 
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El*x
		push!(cache, _permute_malloc(TT, cache[2], ((3,), (1, 2, 4)), TO))
		# [4] Hr * El * x
		push!(cache, _mul_malloc(TT, Hr.A, cache[3], TO))
		# [5] permute Hr * El * x
		push!(cache, _permute_malloc(TT, cache[4], ((2, 3, 1), (4,)), TO))
		# [6] Hr * El * x * Er
		push!(cache, _mul_malloc(TT, cache[5], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x 
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El*x
		_permute_TO!(cache[3], cache[2], ((3,), (1, 2, 4)), TO)
		# Hr * El * x
		_mul_TO!(cache[4], Hr.A, cache[3], TO)
		# permute Hr * El * x
		_permute_TO!(cache[5], cache[4], ((2, 3, 1), (4,)), TO)
		# Hr * El * x * Er
		_mul_TO!(cache[6], cache[5], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[6], ((1, 2), (3, 4)), TO)
	return x

end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::LocalOperator{1, 2},
	Hr::LocalOperator{2, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hl), eltype(Hr), eltype(Er))
	# El x, Hl, Hr, Er
	if isempty(cache)
		# [1] permute x 
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4)), TO))
		# [4] permute Hl 
		push!(cache, _permute_malloc(TT, Hl.A, ((1, 3), (2,)), TO))
		# [5] Hl * El * x
		push!(cache, _mul_malloc(TT, cache[4], cache[3], TO))
		# [6] permute Hl * El * x
		push!(cache, _permute_malloc(TT, cache[5], ((3, 1, 5), (2, 4)), TO))
		# [7] permute Hr
		push!(cache, _permute_malloc(TT, Hr.A, ((1, 3), (2,)), TO))
		# [8] Hl * El * x * Hr
		push!(cache, _mul_malloc(TT, cache[6], cache[7], TO))
		# [9] permute Hl * El * x * Hr
		push!(cache, _permute_malloc(TT, cache[8], ((1, 2, 4), (3,)), TO))
		# [10] Hl * El * x * Hr * Er
		push!(cache, _mul_malloc(TT, cache[9], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x 
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El * x
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3, 4)), TO)
		# Hl * El * x
		_mul_TO!(cache[5], cache[4], cache[3], TO)
		# permute Hl * El * x
		_permute_TO!(cache[6], cache[5], ((3, 1, 5), (2, 4)), TO)
		# Hl * El * x * Hr
		_mul_TO!(cache[8], cache[6], cache[7], TO)
		# permute Hl * El * x * Hr
		_permute_TO!(cache[9], cache[8], ((1, 2, 4), (3,)), TO)
		# Hl * El * x * Hr * Er
		_mul_TO!(cache[10], cache[9], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[10], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::LocalOperator{1, 1},
	Hr::LocalOperator{1, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hl), eltype(Hr), eltype(Er))
	# El x, Hl, Hr, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4)), TO))
		# [4] Hl * El * x
		push!(cache, _mul_malloc(TT, Hl.A, cache[3], TO))
		# [5] permute Hl * El * x
		push!(cache, _permute_malloc(TT, cache[4], ((3,), (2, 1, 4)), TO))
		# [6] Hr * Hl * El * x
		push!(cache, _mul_malloc(TT, Hr.A, cache[5], TO))
		# [7] permute Hr * Hl * El * x
		push!(cache, _permute_malloc(TT, cache[6], ((2, 3, 1), (4,)), TO))
		# [8] Hr * Hl * El * x * Er
		push!(cache, _mul_malloc(TT, cache[7], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El * x
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3, 4)), TO)
		# Hl * El * x
		_mul_TO!(cache[4], Hl.A, cache[3], TO)
		# permute Hl * El * x
		_permute_TO!(cache[5], cache[4], ((3,), (2, 1, 4)), TO)
		# Hr * Hl * El * x
		_mul_TO!(cache[6], Hr.A, cache[5], TO)
		# permute Hr * Hl * El * x
		_permute_TO!(cache[7], cache[6], ((2, 3, 1), (4,)), TO)
		# Hr * Hl * El * x * Er
		_mul_TO!(cache[8], cache[7], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)

	end
	_permute_TO!(x.A, cache[8], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{2},
	Hl::LocalOperator{1, 2},
	Hr::IdentityOperator,
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hl), eltype(Er))
	# El x, Hl, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4)), TO))
		# [4] permute Hl
		push!(cache, _permute_malloc(TT, Hl.A, ((1, 3), (2,)), TO))
		# [5] Hl * El * x
		push!(cache, _mul_malloc(TT, cache[4], cache[3], TO))
		# [6] permute Hl * El * x
		push!(cache, _permute_malloc(TT, cache[5], ((3, 1, 4), (5, 2)), TO))
		# [7] Hl * El * x * Er
		push!(cache, _mul_malloc(TT, cache[6], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El * x
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3, 4)), TO)
		# Hl * El * x
		_mul_TO!(cache[5], cache[4], cache[3], TO)
		# permute Hl * El * x
		_permute_TO!(cache[6], cache[5], ((3, 1, 4), (5, 2)), TO)
		# Hl * El * x * Er
		_mul_TO!(cache[7], cache[6], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[7], ((1, 2), (3, 4)), TO)
	return x

end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{3},
	Hl::IdentityOperator,
	Hr::IdentityOperator,
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	# El x, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] permute El 
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,)), TO))
		# [3] El * x
		push!(cache, _mul_malloc(TT, cache[2], cache[1], TO))
		# [4] permute El * x
		push!(cache, _permute_malloc(TT, cache[3], ((1, 3, 4), (5, 2)), TO))
		# [5] El * x * Er
		push!(cache, _mul_malloc(TT, cache[4], Er.A, Hl.strength[] * Hr.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[3], cache[2], cache[1], TO)
		# permute El * x
		_permute_TO!(cache[4], cache[3], ((1, 3, 4), (5, 2)), TO)
		# El * x * Er
		_mul_TO!(cache[5], cache[4], Er.A, Hl.strength[] * Hr.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[5], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::CompositeMPSTensor{2, T},
	El::LocalLeftTensor{3},
	Hl::IdentityOperator,
	Hr::LocalOperator{2, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing}) where T <: NTuple{2, MPSTensor{3}}

	TT = promote_type(eltype(x), eltype(El), eltype(Hr), eltype(Er))
	# x, Er, Hr, El
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1, 2, 3), (4,)), TO))
		# [2] x * Er
		push!(cache, _mul_malloc(TT, cache[1], Er.A, TO))
		# [3] permute x * Er
		push!(cache, _permute_malloc(TT, cache[2], ((3,), (1, 2, 4)), TO))
		# [4] Hr * x * Er
		push!(cache, _mul_malloc(TT, Hr.A, cache[3], TO))
		# [5] permute Hr * x * Er
		push!(cache, _permute_malloc(TT, cache[4], ((1, 3), (4, 2, 5)), TO))
		# [6] El * Hr * x * Er
		push!(cache, _mul_malloc(TT, El.A, cache[5], Hl.strength[] * Hr.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1, 2, 3), (4,)), TO)
		# x * Er
		_mul_TO!(cache[2], cache[1], Er.A, TO)
		# permute x * Er
		_permute_TO!(cache[3], cache[2], ((3,), (1, 2, 4)), TO)
		# Hr * x * Er
		_mul_TO!(cache[4], Hr.A, cache[3], TO)
		# permute Hr * x * Er
		_permute_TO!(cache[5], cache[4], ((1, 3), (4, 2, 5)), TO)
		# El * Hr * x * Er
		_mul_TO!(cache[6], El.A, cache[5], Hl.strength[] * Hr.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[6], ((1, 2), (3, 4)), TO)
	return x
end

# ======================== 1-site MPS ========================
function _action!(x::MPSTensor{3},
	El::LocalLeftTensor{2},
	H::LocalOperator{1, 2},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# El, x, H, Er 
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3)), TO))
		# [4] permute H 
		push!(cache, _permute_malloc(TT, H.A, ((1, 3), (2,)), TO))
		# [5] H * El * x
		push!(cache, _mul_malloc(TT, cache[4], cache[3], TO))
		# [6] permute H * El * x
		push!(cache, _permute_malloc(TT, cache[5], ((3, 1), (4, 2)), TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1,), (2, 3)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El * x
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3)), TO)
		# H * El * x
		_mul_TO!(cache[5], cache[4], cache[3], TO)
		# permute H * El * x
		_permute_TO!(cache[6], cache[5], ((3, 1), (4, 2)), TO)
	end

	_mul_TO!(x.A, cache[6], Er.A, H.strength[], 0.0, TO)
	return x
end

function _action!(x::MPSTensor{3},
	El::LocalLeftTensor{2},
	H::IdentityOperator,
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	# El, x, Er 
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((1, 2), (3,)), TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1,), (2, 3)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El * x
		_permute_TO!(cache[3], cache[2], ((1, 2), (3,)), TO)
	end

	_mul_TO!(x.A, cache[3], Er.A, H.strength[], 0.0, TO)

	return x
end

function _action!(x::MPSTensor{3},
	El::LocalLeftTensor{2},
	H::LocalOperator{1, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# H, x, El, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((2,), (1, 3)), TO))
		# [2] H * x
		push!(cache, _mul_malloc(TT, H.A, cache[1], TO))
		# [3] permute H * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3)), TO))
		# [4] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[3], TO))
		# [5] permute El * x
		push!(cache, _permute_malloc(TT, cache[4], ((1, 2), (3,)), TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((2,), (1, 3)), TO)
		# H * x
		_mul_TO!(cache[2], H.A, cache[1], TO)
		# permute H * x
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3)), TO)
		# El * x
		_mul_TO!(cache[4], El.A, cache[3], TO)
		# permute El * x
		_permute_TO!(cache[5], cache[4], ((1, 2), (3,)), TO)
	end

	_mul_TO!(x.A, cache[5], Er.A, H.strength[], 0.0, TO)
	return x
end

function _action!(x::MPSTensor{3},
	El::LocalLeftTensor{3},
	H::LocalOperator{2, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	# x, Er, H, El
	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	if isempty(cache)
		# [1] x * Er
		push!(cache, _mul_malloc(TT, x.A, Er.A, TO))
		# [2] permute x * Er
		push!(cache, _permute_malloc(TT, cache[1], ((2,), (1, 3)), TO))
		# [3] H * x * Er
		push!(cache, _mul_malloc(TT, H.A, cache[2], TO))
		# [4] permute H * x * Er
		push!(cache, _permute_malloc(TT, cache[3], ((1, 3), (2, 4)), TO))
		# [5] El * x * Er
		push!(cache, _mul_malloc(TT, El.A, cache[4], H.strength[], TO))
	else
		# x * Er
		_mul_TO!(cache[1], x.A, Er.A, TO)
		# permute x * Er
		_permute_TO!(cache[2], cache[1], ((2,), (1, 3)), TO)
		# H * x * Er
		_mul_TO!(cache[3], H.A, cache[2], TO)
		# permute H * x * Er
		_permute_TO!(cache[4], cache[3], ((1, 3), (2, 4)), TO)
		# El * x * Er
		_mul_TO!(cache[5], El.A, cache[4], H.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[5], ((1, 2), (3,)), TO)
	return x
end

function _action!(x::MPSTensor{3},
	El::LocalLeftTensor{3},
	H::LocalOperator{1, 1},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# H, x, El, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((2,), (1, 3)), TO))
		# [2] H * x
		push!(cache, _mul_malloc(TT, H.A, cache[1], TO))
		# [3] permute H * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3)), TO))
		# [4] permute El 
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,)), TO))
		# [5] El * H * x
		push!(cache, _mul_malloc(TT, cache[4], cache[3], TO))
		# [6] permute El * H * x
		push!(cache, _permute_malloc(TT, cache[5], ((1, 3), (4, 2)), TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((2,), (1, 3)), TO)
		# H * x
		_mul_TO!(cache[2], H.A, cache[1], TO)
		# permute H * x
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3)), TO)
		# El * H * x
		_mul_TO!(cache[5], cache[4], cache[3], TO)
		# permute El * H * x
		_permute_TO!(cache[6], cache[5], ((1, 3), (4, 2)), TO)
	end

	_mul_TO!(x.A, cache[6], Er.A, H.strength[], 0.0, TO)
	return x
end

function _action!(x::MPSTensor{3},
	El::LocalLeftTensor{3},
	H::IdentityOperator,
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	# El, x, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3)), TO))
		# [2] permute El
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,)), TO))
		# [3] El * x
		push!(cache, _mul_malloc(TT, cache[2], cache[1], TO))
		# [4] permute El * x
		push!(cache, _permute_malloc(TT, cache[3], ((1, 3), (4, 2)), TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1,), (2, 3)), TO)
		# El * x
		_mul_TO!(cache[3], cache[2], cache[1], TO)
		# permute El * x
		_permute_TO!(cache[4], cache[3], ((1, 3), (4, 2)), TO)
	end

	_mul_TO!(x.A, cache[4], Er.A, H.strength[], 0.0, TO)
	return x
end

function _action!(x::MPSTensor{3},
	El::LocalLeftTensor{3},
	H::LocalOperator{2, 2},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{Nothing, TimerOutput})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# x, Er, H, El
	if isempty(cache)
		# [1] permute Er
		push!(cache, _permute_malloc(TT, Er.A, ((1,), (2, 3)), TO))
		# [2] x * Er 
		push!(cache, _mul_malloc(TT, x.A, cache[1], TO))
		# [3] permute x * Er
		push!(cache, _permute_malloc(TT, cache[2], ((2, 3), (1, 4)), TO))
		# [4] H * x * Er
		push!(cache, _mul_malloc(TT, H.A, cache[3], TO))
		# [5] permute H * x * Er
		push!(cache, _permute_malloc(TT, cache[4], ((1, 3), (2, 4)), TO))
		# [6] El * H * x * Er
		push!(cache, _mul_malloc(TT, El.A, cache[5], H.strength[], TO))
	else
		# x * Er
		_mul_TO!(cache[2], x.A, cache[1], TO)
		# permute x * Er
		_permute_TO!(cache[3], cache[2], ((2, 3), (1, 4)), TO)
		# H * x * Er
		_mul_TO!(cache[4], H.A, cache[3], TO)
		# permute H * x * Er
		_permute_TO!(cache[5], cache[4], ((1, 3), (2, 4)), TO)
		# El * H * x * Er
		_mul_TO!(cache[6], El.A, cache[5], H.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[6], ((1, 2), (3,)), TO)
	return x
end

# ======================== 1-site MPO ========================
function _action!(x::MPSTensor{4},
	El::LocalLeftTensor{2},
	H::LocalOperator{1, 2},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# El, x, H, Er 
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4)), TO))
		# [4] permute H 
		push!(cache, _permute_malloc(TT, H.A, ((1, 3), (2,)), TO))
		# [5] H * El * x
		push!(cache, _mul_malloc(TT, cache[4], cache[3], TO))
		# [6] permute H * El * x
		push!(cache, _permute_malloc(TT, cache[5], ((3, 1, 4), (5, 2)), TO))
		# [7] H * El * x * Er
		push!(cache, _mul_malloc(TT, cache[6], Er.A, H.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El * x
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3, 4)), TO)
		# H * El * x
		_mul_TO!(cache[5], cache[4], cache[3], TO)
		# permute H * El * x
		_permute_TO!(cache[6], cache[5], ((3, 1, 4), (5, 2)), TO)
		# H * El * x * Er
		_mul_TO!(cache[7], cache[6], Er.A, H.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[7], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::MPSTensor{4},
	El::LocalLeftTensor{2},
	H::IdentityOperator,
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	# El, x, Er 
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[1], TO))
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((1, 2, 3), (4,)), TO))
		# [4] El * x * Er
		push!(cache, _mul_malloc(TT, cache[3], Er.A, H.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[2], El.A, cache[1], TO)
		# permute El * x
		_permute_TO!(cache[3], cache[2], ((1, 2, 3), (4,)), TO)
		# El * x * Er
		_mul_TO!(cache[4], cache[3], Er.A, H.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[4], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::MPSTensor{4},
	El::LocalLeftTensor{2},
	H::LocalOperator{1, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# H, x, El, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((2,), (1, 3, 4)), TO))
		# [2] H * x
		push!(cache, _mul_malloc(TT, H.A, cache[1], TO))
		# [3] permute H * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4)), TO))
		# [4] El * x
		push!(cache, _mul_malloc(TT, El.A, cache[3], TO))
		# [5] permute El * x
		push!(cache, _permute_malloc(TT, cache[4], ((1, 2, 3), (4,)), TO))
		# [6] El * x * Er
		push!(cache, _mul_malloc(TT, cache[5], Er.A, H.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((2,), (1, 3, 4)), TO)
		# H * x
		_mul_TO!(cache[2], H.A, cache[1], TO)
		# permute H * x
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[4], El.A, cache[3], TO)
		# permute El * x
		_permute_TO!(cache[5], cache[4], ((1, 2, 3), (4,)), TO)
		# El * x * Er
		_mul_TO!(cache[6], cache[5], Er.A, H.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[6], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::MPSTensor{4},
	El::LocalLeftTensor{3},
	H::LocalOperator{2, 1},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	# x, Er, H, El
	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1, 2, 3), (4,)), TO))
		# [2] x * Er
		push!(cache, _mul_malloc(TT, cache[1], Er.A, TO))
		# [3] permute x * Er
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4)), TO))
		# [4] H * x * Er
		push!(cache, _mul_malloc(TT, H.A, cache[3], TO))
		# [5] permute H * x * Er
		push!(cache, _permute_malloc(TT, cache[4], ((1, 3), (2, 4, 5)), TO))
		# [6] El * x * Er
		push!(cache, _mul_malloc(TT, El.A, cache[5], H.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1, 2, 3), (4,)), TO)
		# x * Er
		_mul_TO!(cache[2], cache[1], Er.A, TO)
		# permute x * Er
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3, 4)), TO)
		# H * x * Er
		_mul_TO!(cache[4], H.A, cache[3], TO)
		# permute H * x * Er
		_permute_TO!(cache[5], cache[4], ((1, 3), (2, 4, 5)), TO)
		# El * x * Er
		_mul_TO!(cache[6], El.A, cache[5], H.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[6], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::MPSTensor{4},
	El::LocalLeftTensor{3},
	H::LocalOperator{1, 1},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# H, x, El, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((2,), (1, 3, 4)), TO))
		# [2] H * x
		push!(cache, _mul_malloc(TT, H.A, cache[1], TO))
		# [3] permute H * x
		push!(cache, _permute_malloc(TT, cache[2], ((2,), (1, 3, 4)), TO))
		# [4] permute El 
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,)), TO))
		# [5] El * H * x
		push!(cache, _mul_malloc(TT, cache[4], cache[3], TO))
		# [6] permute El * H * x
		push!(cache, _permute_malloc(TT, cache[5], ((1, 3, 4), (5, 2)), TO))
		# [7] El * H * x * Er
		push!(cache, _mul_malloc(TT, cache[6], Er.A, H.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((2,), (1, 3, 4)), TO)
		# H * x
		_mul_TO!(cache[2], H.A, cache[1], TO)
		# permute H * x
		_permute_TO!(cache[3], cache[2], ((2,), (1, 3, 4)), TO)
		# El * H * x
		_mul_TO!(cache[5], cache[4], cache[3], TO)
		# permute El * H * x
		_permute_TO!(cache[6], cache[5], ((1, 3, 4), (5, 2)), TO)
		# El * H * x * Er
		_mul_TO!(cache[7], cache[6], Er.A, H.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[7], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::MPSTensor{4},
	El::LocalLeftTensor{3},
	H::IdentityOperator,
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	# El, x, Er
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1,), (2, 3, 4)), TO))
		# [2] permute El
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,)), TO))
		# [3] El * x
		push!(cache, _mul_malloc(TT, cache[2], cache[1], TO))
		# [4] permute El * x
		push!(cache, _permute_malloc(TT, cache[3], ((1, 3, 4), (5, 2)), TO))
		# [5] El * x * Er
		push!(cache, _mul_malloc(TT, cache[4], Er.A, H.strength[], TO))
	else
		# permute x
		_permute_TO!(cache[1], x.A, ((1,), (2, 3, 4)), TO)
		# El * x
		_mul_TO!(cache[3], cache[2], cache[1], TO)
		# permute El * x
		_permute_TO!(cache[4], cache[3], ((1, 3, 4), (5, 2)), TO)
		# El * x * Er
		_mul_TO!(cache[5], cache[4], Er.A, H.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[5], ((1, 2), (3, 4)), TO)
	return x
end

function _action!(x::MPSTensor{4}, El::LocalLeftTensor{3}, H::LocalOperator{2, 2}, Er::LocalRightTensor{3}, cache::Vector{<:AbstractTensorMap}, TO::Union{Nothing, TimerOutput})

	TT = promote_type(eltype(x), eltype(El), eltype(H), eltype(Er))
	# x, Er, H, El
	if isempty(cache)
		# [1] permute x
		push!(cache, _permute_malloc(TT, x.A, ((1, 2, 3), (4,)), TO))
		# [2] permute Er
		push!(cache, _permute_malloc(TT, Er.A, ((1,), (2, 3)), TO))
		# [3] x * Er 
		push!(cache, _mul_malloc(TT, cache[1], cache[2], TO))
		# [4] permute x * Er
		push!(cache, _permute_malloc(TT, cache[3], ((2, 4), (1, 3, 5)), TO))
		# [5] H * x * Er
		push!(cache, _mul_malloc(TT, H.A, cache[4], TO))
		# [6] permute H * x * Er
		push!(cache, _permute_malloc(TT, cache[5], ((1, 3), (2, 4, 5)), TO))
		# [7] El * H * x * Er
		push!(cache, _mul_malloc(TT, El.A, cache[6], H.strength[], TO))
	else
		# permute x 
		_permute_TO!(cache[1], x.A, ((1, 2, 3), (4,)), TO)
		# x * Er
		_mul_TO!(cache[3], cache[1], cache[2], TO)
		# permute x * Er
		_permute_TO!(cache[4], cache[3], ((2, 4), (1, 3, 5)), TO)
		# H * x * Er
		_mul_TO!(cache[5], H.A, cache[4], TO)
		# permute H * x * Er
		_permute_TO!(cache[6], cache[5], ((1, 3), (2, 4, 5)), TO)
		# El * H * x * Er
		_mul_TO!(cache[7], El.A, cache[6], H.strength[], 0.0, TO)
	end

	_permute_TO!(x.A, cache[7], ((1, 2), (3, 4)), TO)
	return x
end

# ======================== bond ========================
function _action!(x::MPSTensor{2},
	El::LocalLeftTensor{3},
	Er::LocalRightTensor{3},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	# El, x, Er
	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	if isempty(cache)
		# [1] permute El
		push!(cache, _permute_malloc(TT, El.A, ((1, 2), (3,)), TO))
		# [2] El * x 
		push!(cache, _mul_malloc(TT, cache[1], x.A, TO))
		# [3] permute El * x
		push!(cache, _permute_malloc(TT, cache[2], ((1,), (3, 2)), TO))
	else
		# El * x 
		_mul_TO!(cache[2], cache[1], x.A, TO)
		# permute El * x
		_permute_TO!(cache[3], cache[2], ((1,), (3, 2)), TO)
	end

	# El * x * Er
	_mul_TO!(x.A, cache[3], Er.A, TO)
	return x
end
function _action!(x::MPSTensor{2},
	El::LocalLeftTensor{2},
	Er::LocalRightTensor{2},
	cache::Vector{<:AbstractTensorMap},
	TO::Union{TimerOutput, Nothing})

	TT = promote_type(eltype(x), eltype(El), eltype(Er))
	# El, x, Er
	if isempty(cache)
		# [1] El * x
		push!(cache, _mul_malloc(TT, El.A, x.A, TO))
	else
		# El * x 
		_mul_TO!(cache[1], El.A, x.A, TO)
	end

	# El * x * Er
	_mul_TO!(x.A, cache[1], Er.A, TO)
	return x
end


# ======================== utils ========================
function _permute_malloc(
	T::Type{<:Union{Float64, ComplexF64}},
	A::AbstractTensorMap,
	pA::Index2Tuple,
	::Nothing)
	# alloc a new tensor with manual allocator
	t = tensoralloc_add(T, A, pA, false, Val(true), ManualAllocator())
	permute!(t, A, pA)
	return t
end
function _permute_malloc(
	T::Type{<:Union{Float64, ComplexF64}},
	A::AbstractTensorMap,
	pA::Index2Tuple,
	TO::TimerOutput)
	# alloc a new tensor with manual allocator
	@timeit TO "malloc" t = tensoralloc_add(T, A, pA, false, Val(true), ManualAllocator())
	@timeit TO "permute" permute!(t, A, pA)
	return t
end


function _mul_malloc(
	T::Type{<:Union{Float64, ComplexF64}},
	A::AbstractTensorMap,
	B::AbstractTensorMap,
	α::Number,
	::Nothing)
	# α * A * B with manual allocator
	s = HomSpace(codomain(A), domain(B))
	d = fusionblockstructure(s).totaldim
	t = TensorMap{T}(tensoralloc(Vector{T}, d, Val(true), ManualAllocator()), s)
	rmul!(t, 0.0)
	_mul_TO!(t, A, B, α, 0.0, nothing)
	return t
end
function _mul_malloc(
	T::Type{<:Union{Float64, ComplexF64}},
	A::AbstractTensorMap,
	B::AbstractTensorMap,
	α::Number,
	TO::TimerOutput)
	# α * A * B with manual allocator
	s = HomSpace(codomain(A), domain(B))
	d = fusionblockstructure(s).totaldim
	@timeit TO "malloc" t = TensorMap{T}(tensoralloc(Vector{T}, d, Val(true), ManualAllocator()), s)
	rmul!(t, 0.0)
	_mul_TO!(t, A, B, α, 0.0, TO)
	return t
end
# default α = 1.0
_mul_malloc(C::Type{<:Union{Float64, ComplexF64}}, A::AbstractTensorMap, B::AbstractTensorMap, TO::Union{TimerOutput, Nothing}) = _mul_malloc(C, A, B, 1.0, TO)

function _permute_TO!(C::AbstractTensorMap, A::AbstractTensorMap, pA::Index2Tuple, TO::TimerOutput)
	@timeit TO "permute" permute!(C, A, pA)
	return C
end
_permute_TO!(C::AbstractTensorMap, A::AbstractTensorMap, pA::Index2Tuple, ::Nothing) = permute!(C, A, pA)

function _mul_TO!(C::AbstractTensorMap, A::AbstractTensorMap, B::AbstractTensorMap, α::Number, β::Number, ::Nothing)
	mul!(C, A, B, α, β)
	# r1, r2, r3 = numout(A), numin(A), numin(B)
	# pA = (Tuple(1:r1), Tuple(r1 + 1:r1+r2))
	# pB = (Tuple(1:r2), Tuple(r2 + 1:r2+r3))
	# pC = (Tuple(1:r1), Tuple(r1 + 1:r1+r3))
	# tensorcontract!(C, A, pA, false, B, pB, false, pC, α, β)
	return C
end
function _mul_TO!(C::AbstractTensorMap, A::AbstractTensorMap, B::AbstractTensorMap, α::Number, β::Number, TO::TimerOutput)
	@timeit TO "contract" _mul_TO!(C, A, B, α, β, nothing)
	return C
end
# default α = 1.0, β = 0.0
_mul_TO!(C::AbstractTensorMap, A::AbstractTensorMap, B::AbstractTensorMap, α::Number, TO::Union{TimerOutput, Nothing}) = _mul_TO!(C, A, B, α, 0.0, TO)
_mul_TO!(C::AbstractTensorMap, A::AbstractTensorMap, B::AbstractTensorMap, TO::Union{TimerOutput, Nothing}) = _mul_TO!(C, A, B, 1.0, 0.0, TO)

tensorfree!(x::Array{<:Union{Float64, ComplexF64}}, ::ManualAllocator) = nothing

function _action_str(PH::SimpleProjectiveHamiltonian)
	return join([_action_str(PH.El), _action_str.(PH.H)..., _action_str(PH.Er)], "_")
end
_action_str(::LocalLeftTensor{R}) where R = "$(R)"
_action_str(::LocalOperator{R1, R2}) where {R1, R2} = "$(R1)$(R2)"
_action_str(::IdentityOperator) = "00"
_action_str(::LocalRightTensor{R}) where R = "$(R)"