struct LeftOrthComplement{N}
	El::Vector{AbstractTensorMap}
	Al::Vector{AbstractTensorMap}
	Al_c::AbstractTensorMap

	function LeftOrthComplement(El::Vector{AbstractTensorMap}, Al::Vector{AbstractTensorMap}, Al_c::AbstractTensorMap)
	     # directly construct
	     N = length(El)
	     return new{N}(El, Al, Al_c)
	end
	function LeftOrthComplement(El_i::SparseLeftTensor, Al_c::MPSTensor{R}, Hl::SparseMPOTensor, Al_i::MPSTensor{R} = Al_c) where {R}
		# initialize Al
		Al = _initialize_Al(El_i, Al_i, Hl)
		# initialize El
		tAl_c = permute(Al_c.A, (Tuple(1:R-1), (R,)))
		El = _initialize_El(Al, tAl_c)
		N = length(Al)
		return new{N}(El, Al, tAl_c)
	end
end

struct RightOrthComplement{N}
	Er::Vector{AbstractTensorMap}
	Ar::Vector{AbstractTensorMap}
	Ar_c::AbstractTensorMap
	function RightOrthComplement(Er::Vector{AbstractTensorMap}, Ar::Vector{AbstractTensorMap}, Ar_c::AbstractTensorMap)
		# directly construct
		N = length(Er)
		return new{N}(Er, Ar, Ar_c)
	end
	function RightOrthComplement(Er_i::SparseRightTensor, Ar_c::MPSTensor{R}, Hr::SparseMPOTensor, Ar_i::MPSTensor{R} = Ar_c) where {R}
		# initialize Ar
		Ar = _initialize_Ar(Er_i, Ar_i, Hr)
		# initialize Er
		tAr_c = permute(Ar_c.A, ((1,), Tuple(2:R)))
		Er = _initialize_Er(Ar, tAr_c)
		N = length(Ar)
		return new{N}(Er, Ar, tAr_c)
	end
end

length(::LeftOrthComplement{N}) where {N} = N
length(::RightOrthComplement{N}) where {N} = N

function _initialize_Al(El_i::SparseLeftTensor, Al_i::MPSTensor, Hl::SparseMPOTensor)

	sz = size(Hl)
	Al = Vector{AbstractTensorMap}(undef, sz[2])

	validIdx = filter!(x -> !isnothing(El_i[x[1]]) && !isnothing(Hl[x[1], x[2]]), [(i, j) for i in 1:sz[1] for j in 1:sz[2]])

	if get_num_workers() > 1
		lsAl = pmap(validIdx) do (i, j)
			_initialize_Al_single(El_i[i], Al_i, Hl[i, j])
		end

		# sum over i
		for (idx, (_, j)) in enumerate(validIdx)
			if !isassigned(Al, j)
				Al[j] = lsAl[idx]
			else
				axpy!(true, lsAl[idx], Al[j])
			end
		end

	else
		# TODO sort by cost 
		Lock = Threads.ReentrantLock()
		Threads.@threads :greedy for (i, j) in validIdx
			tmp = _initialize_Al_single(El_i[i], Al_i, Hl[i, j])
			lock(Lock)
			try
				if !isassigned(Al, j)
					Al[j] = tmp
				else
					add!(Al[j], tmp)
				end
			catch
				rethrow()
			finally
				unlock(Lock)
			end

		end

	end

	return Al
end

function _initialize_Ar(Er_i::SparseRightTensor, Ar_i::MPSTensor, Hr::SparseMPOTensor)

	sz = size(Hr)
	Ar = Vector{AbstractTensorMap}(undef, sz[1])

	validIdx = filter!(x -> !isnothing(Er_i[x[2]]) && !isnothing(Hr[x[1], x[2]]), [(i, j) for i in 1:sz[1] for j in 1:sz[2]])

	if get_num_workers() > 1
		lsAr = pmap(validIdx) do (i, j)
			_initialize_Ar_single(Er_i[j], Ar_i, Hr[i, j])
		end
		# sum over j
		for (idx, (i, j)) in enumerate(validIdx)
			if !isassigned(Ar, i)
				Ar[i] = lsAr[idx]
			else
				axpy!(true, lsAr[idx], Ar[i])
			end
		end

	else

		Lock = Threads.ReentrantLock()
		Threads.@threads :greedy for (i, j) in validIdx
			tmp = _initialize_Ar_single(Er_i[j], Ar_i, Hr[i, j])
			lock(Lock)
			try
				if !isassigned(Ar, i)
					Ar[i] = tmp
				else
					add!(Ar[i], tmp)
				end
			catch
				rethrow()
			finally
				unlock(Lock)
			end

		end

	end

	return Ar
end

function _initialize_El(Al::Vector{AbstractTensorMap}, Al_i::AbstractTensorMap)

	sz = length(Al)

	if get_num_workers() > 1
		return pmap(Al) do Al
			_initialize_El_single(Al, Al_i)
		end
	else
		El = Vector{AbstractTensorMap}(undef, sz)

		Threads.@threads :greedy for i in 1:sz
			El[i] = Al_i' * Al[i]
		end
		return El
	end

end

function _initialize_Er(Ar::Vector{AbstractTensorMap}, Ar_i::AbstractTensorMap)

	sz = length(Ar)

	if get_num_workers() > 1
		return pmap(Ar) do Ar
			_initialize_Er_single(Ar, Ar_i)
		end
	else
		Er = Vector{AbstractTensorMap}(undef, sz)

		Threads.@threads :greedy for i in 1:sz
               Er[i] = Ar[i] * Ar_i'
		end

		return Er
	end
end

# =================== contract El, Al and Hl ====================
#             e(3)
#             |
#    --- c ---Al--- f(5)     
#   |         |
#   |         d
#   |         |
#  El--- b ---Hl--- h(4)
#   |         |
#   |         g(2)
#   |
#    --- a(1)
function _initialize_Al_single(El::LocalLeftTensor{2}, Al::MPSTensor{R}, Hl::IdentityOperator) where R
	Al_f = El.A * permute(Al.A, ((1,), Tuple(2:R)))
	return rmul!(permute(Al_f, (Tuple(1:R-1), (R,))), Hl.strength[])
end
function _initialize_Al_single(El::LocalLeftTensor{2}, Al::MPSTensor{4}, Hl::LocalOperator{1, 1})
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a g e; f] := s * El.A[a c] * (Hl.A[g d] * Al.A[c d e f])
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{2}, Al::MPSTensor{3}, Hl::LocalOperator{1, 1})
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a g; f] := s * El.A[a c] * (Hl.A[g d] * Al.A[c d f])
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{2}, Al::MPSTensor{4}, Hl::LocalOperator{1, 2})
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a g e; h f] := s * (El.A[a c] * Al.A[c d e f]) * Hl.A[g d h]
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{2}, Al::MPSTensor{3}, Hl::LocalOperator{1, 2})
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a g; h f] := s * (El.A[a c] * Al.A[c d f]) * Hl.A[g d h]
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{4}, Hl::IdentityOperator)
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a d e; b f] := s * El.A[a b c] * Al.A[c d e f]
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{3}, Hl::IdentityOperator)
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a d; b f] := s * El.A[a b c] * Al.A[c d f]
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{4}, Hl::LocalOperator{2, 1})
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a g e; f] := s * (El.A[a b c] * Al.A[c d e f]) * Hl.A[b g d]
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{3}, Hl::LocalOperator{2, 1})
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a g; f] := s * (El.A[a b c] * Al.A[c d f]) * Hl.A[b g d]
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{4}, Hl::LocalOperator{1, 1})
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a g e; b f] := s * El.A[a b c] * (Hl.A[g d] * Al.A[c d e f])
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{3}, Hl::LocalOperator{1, 1})
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a g; b f] := s * El.A[a b c] * (Hl.A[g d] * Al.A[c d f])
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{4}, Hl::LocalOperator{2, 2})
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a g e; h f] := s * (El.A[a b c] * Al.A[c d e f]) * Hl.A[b g d h]
	return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{3}, Hl::LocalOperator{2, 2})
	s = Hl.strength[]
	@tensor allocator = ManualAllocator() tmp[a g; h f] := s * (El.A[a b c] * Al.A[c d f]) * Hl.A[b g d h]
	return tmp
end
# ------------------------------------------------------

# =================== contract Er, Ar and Hr ====================
#          c(4)
#           |
#   a(2) ---Ar--- d ---    
#           |          |
#           b          |
#           |          |
#   e(1) ---Hr--- g ---Er
#           |          |
#          f(3)        |
#                      |
#              h(5) ---
function _initialize_Ar_single(Er::LocalRightTensor{2}, Ar::MPSTensor{R}, Hr::IdentityOperator) where R
     Ar_f = permute(Ar.A, (Tuple(1:R-1), (R,))) * Er.A
	return rmul!(permute(Ar_f, ((1,), Tuple(2:R))), Hr.strength[])
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{4}, Hr::IdentityOperator)
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[g a; b c h] := s * Ar.A[a b c d] * Er.A[d g h]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{3}, Hr::IdentityOperator)
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[g a; b h] := s * Ar.A[a b d] * Er.A[d g h]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{2}, Ar::MPSTensor{4}, Hr::LocalOperator{1, 1})
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[a; f c h] := s * (Ar.A[a b c d] * Er.A[d h]) * Hr.A[f b]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{2}, Ar::MPSTensor{3}, Hr::LocalOperator{1, 1})
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[a; f h] := s * (Ar.A[a b d] * Er.A[d h]) * Hr.A[f b]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{2}, Ar::MPSTensor{4}, Hr::LocalOperator{2, 1})
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[e a; f c h] := s * (Ar.A[a b c d] * Er.A[d h]) * Hr.A[e f b]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{2}, Ar::MPSTensor{3}, Hr::LocalOperator{2, 1})
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[e a; f h] := s * (Ar.A[a b d] * Er.A[d h]) * Hr.A[e f b]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{4}, Hr::LocalOperator{1, 1})
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[g a; f c h] := s * (Hr.A[f b] * Ar.A[a b c d]) * Er.A[d g h]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{3}, Hr::LocalOperator{1, 1})
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[g a; f h] := s * (Hr.A[f b] * Ar.A[a b d]) * Er.A[d g h]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{4}, Hr::LocalOperator{1, 2})
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[a; f c h] := s * (Hr.A[f b g] * Ar.A[a b c d]) * Er.A[d g h]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{3}, Hr::LocalOperator{1, 2})::MPSTensor
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[a; f h] := s * (Hr.A[f b g] * Ar.A[a b d]) * Er.A[d g h]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{4}, Hr::LocalOperator{2, 2})
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[e a; f c h] := s * (Ar.A[a b c d] * Er.A[d g h]) * Hr.A[e f b g]
	return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{3}, Hr::LocalOperator{2, 2})
	s = Hr.strength[]
	@tensor allocator = ManualAllocator() tmp[e a; f h] := s * (Ar.A[a b d] * Er.A[d g h]) * Hr.A[e f b g]
	return tmp
end
# # =================== contract Al and Al_i ====================
# #      c
# #      | 
# #   a--Al--e(3) 
# #      | \  
# #      b  d(2)
# #      |
# #  a--Al_i'--f(1)
# #      |
# #      c
# function _initialize_El_single(Al::MPSTensor{4}, Al_i::MPSTensor{4})::LocalLeftTensor
# 	@tensor allocator = ManualAllocator() tmp[f; e] := Al.A[a b c e] * Al_i.A'[c f a b]
# 	return tmp
# end
# function _initialize_El_single(Al::MPSTensor{3}, Al_i::MPSTensor{3})::LocalLeftTensor
# 	@tensor allocator = ManualAllocator() tmp[f; e] := Al.A[a b e] * Al_i.A'[f a b]
# 	return tmp
# end
# function _initialize_El_single(Al::MPSTensor{5}, Al_i::MPSTensor{4})::LocalLeftTensor
# 	@tensor allocator = ManualAllocator() tmp[f d; e] := Al.A[a b c d e] * Al_i.A'[c f a b]
# 	return tmp
# end
# function _initialize_El_single(Al::MPSTensor{4}, Al_i::MPSTensor{3})::LocalLeftTensor
# 	@tensor allocator = ManualAllocator() tmp[f d; e] := Al.A[a b d e] * Al_i.A'[f a b]
# 	return tmp
# end
# # --------------------------------------------------------------

# # =================== contract Ar and Ar_i ====================
# #         c
# #         | 
# #  a(1)---Ar---e 
# #       / |   
# #   d(2)  b  
# #         |
# #  f(3)--Ar_i'--e
# #         |
# #         c
# function _initialize_Er_single(Ar::MPSTensor{4}, Ar_i::MPSTensor{4})::LocalRightTensor
# 	@tensor allocator = ManualAllocator() tmp[a; f] := Ar.A[a b c e] * Ar_i.A'[c e f b]
# 	return tmp
# end
# function _initialize_Er_single(Ar::MPSTensor{3}, Ar_i::MPSTensor{3})::LocalRightTensor
# 	@tensor allocator = ManualAllocator() tmp[a; f] := Ar.A[a b e] * Ar_i.A'[e f b]
# 	return tmp
# end
# function _initialize_Er_single(Ar::MPSTensor{5}, Ar_i::MPSTensor{4})::LocalRightTensor
# 	@tensor allocator = ManualAllocator() tmp[a; d f] := Ar.A[a b c d e] * Ar_i.A'[c e f b]
# 	return tmp
# end
# function _initialize_Er_single(Ar::MPSTensor{4}, Ar_i::MPSTensor{3})::LocalRightTensor
# 	@tensor allocator = ManualAllocator() tmp[a; d f] := Ar.A[a b d e] * Ar_i.A'[e f b]
# 	return tmp
# end
# # --------------------------------------------------------------
