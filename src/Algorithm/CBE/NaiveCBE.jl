function _CBE(Al::MPSTensor{R1}, Ar::MPSTensor{R2}, El::SparseLeftTensor, Er::SparseRightTensor, Hl::SparseMPOTensor, Hr::SparseMPOTensor, Alg::NaiveCBE{SweepR2L}, TO::TimerOutput;
	Bl::MPSTensor{R1} = Al,
	Br::MPSTensor{R2} = Ar,
	) where {R1, R2}
	# Al is left canonical, Ar is canonical center
	# Bl, Br are top layer tensors

	# right canonical Ar 
	Ar_perm = permute(Ar.A, ((1,), Tuple(2:R2)))
	@timeit TO "qr" _, Ar_c::MPSTensor = rightorth(Ar_perm)

	# construct L/R orth complement obj 
	@timeit TO "construct LO" LO = LeftOrthComplement(El, Al, Hl, Bl) |> _orth!
	@timeit TO "construct RO" RO = RightOrthComplement(Er, Ar_c, Hr, Br) |> _orth!

	# svd
	D_add = Alg.D - dim(Al, R2)[2]
	if Alg.rsvd
		# reduce the LO * RO complexity from O(D^3d^2) to O(D^3d), svd complexity form O(D^3d^3) to O(D^3d)
		# 1. LO_trunc = Ω * LO, O(D^3dχ)
		# 2. x2_trunc = LO_trunc * RO, O(D^3dχ)
		# 3. _, Q = qr(x2_trunc), O(D^3d)
		# 4. RO_trunc = RO * Q', O(D^3dχ)
		# 5. x2 = LO * RO_trunc, O(D^3dχ)
		# 6. svd, O(D^3d)
		V_trunc = _rsvd_trunc(codomain(LO.Al_c), Alg.D)
		Ω = randn(eltype(LO.Al_c), V_trunc, codomain(LO.Al_c))
          @timeit TO "truncate LO" LO_trunc = _apply_Ω(LO, Ω)
          @timeit TO "LO_trunc * RO" x2_trunc = _contractLR!(LO_trunc, RO)
          @timeit TO "qr" _, Q = rightorth!(x2_trunc)
          @timeit TO "truncate RO" RO_trunc = _apply_Ω(RO, Q')
          @timeit TO "LO * RO_trunc" x2 = _contractLR!(LO, RO_trunc)
          @timeit TO "svd" Al_ex, s, _, ϵ = _tsvd_try(x2; trunc = truncdim(D_add) & truncbelow(Alg.tol))
          Ar_ex = zeros(eltype(Ar_perm), domain(Al_ex), domain(Ar_perm))
          ϵp = NaN
	else
		# contract LO and RO
		@timeit TO "LO * RO" x2 = _contractLR!(LO, RO)
		ϵp = norm(x2) # this can be an estimation of the projection error

		# normal svd, O(D^3d^3)
		@timeit TO "svd" Al_ex, s, Ar_ex, ϵ = _tsvd_try(x2; trunc = truncdim(D_add) & truncbelow(Alg.tol))
		rmul!(Ar_ex, 0.0)
	end
	info = BondInfo(s, ϵ)

	# oplus
	@timeit TO "oplus" begin
		Al_f = catdomain(LO.Al_c, Al_ex)
		Ar_f = catcodomain(Ar_perm, Ar_ex)
	end

	if Alg.check
		@timeit TO "check" ϵ = norm(Al_f * Ar_f - LO.Al_c * Ar_perm)
	else
		ϵ = NaN
	end

	return Al_f, Ar_f, CBEInfo(Alg, (info,), dim(Ar, 1), dim(Ar_f, 1), ϵp, ϵ)
end

function _CBE(Al::MPSTensor{R1}, Ar::MPSTensor{R2}, El::SparseLeftTensor, Er::SparseRightTensor, Hl::SparseMPOTensor, Hr::SparseMPOTensor, Alg::NaiveCBE{SweepL2R}, TO::TimerOutput;
	Bl::MPSTensor{R1} = Al,
	Br::MPSTensor{R2} = Ar,
	) where {R1, R2}
	# Al is canonical center, Ar is right canonical

	# left canonical Al
	Al_perm = permute(Al.A, (Tuple(1:R1-1), (R1,)))
	@timeit TO "qr" Al_c::MPSTensor, _ = leftorth(Al_perm)

	# construct L/R orth complement obj
	@timeit TO "construct LO" LO = LeftOrthComplement(El, Al_c, Hl, Bl) |> _orth!
	@timeit TO "construct RO" RO = RightOrthComplement(Er, Ar, Hr,  Br) |> _orth!

	D_add = Alg.D - dim(Ar, 1)[2]
	if Alg.rsvd
		# reduce the LO * RO complexity from O(D^3d^2) to O(D^3d), svd complexity form O(D^3d^3) to O(D^3d)
		# 1. RO_trunc = RO * Ω, O(D^3dχ)
		# 2. x2_trunc = LO * RO_trunc, O(D^3dχ)
		# 3. Q, _ = qr(x2_trunc), O(D^3d)
		# 4. LO_trunc = Q' * LO, O(D^3dχ)
		# 5. x2 = LO_trunc * RO, O(D^3dχ)
		# 6. svd, O(D^3d)
		V_trunc = _rsvd_trunc(domain(RO.Ar_c), Alg.D)
		Ω = randn(eltype(RO.Ar_c), domain(RO.Ar_c), V_trunc)
		@timeit TO "truncate RO" RO_trunc = _apply_Ω(RO, Ω)
		@timeit TO "LO * RO_trunc" x2_trunc = _contractLR!(LO, RO_trunc)
		@timeit TO "qr" Q, _ = leftorth!(x2_trunc)
		@timeit TO "truncate LO" LO_trunc = _apply_Ω(LO, Q')
		@timeit TO "LO_trunc * RO" x2 = _contractLR!(LO_trunc, RO)
		@timeit TO "svd" _, s, Ar_ex, ϵ = _tsvd_try(x2; trunc = truncdim(D_add) & truncbelow(Alg.tol))
		Al_ex = zeros(eltype(Al_perm), codomain(Al_perm), codomain(Ar_ex))
		ϵp = NaN
	else
		# contract LO and RO
		@timeit TO "LO * RO" x2 = _contractLR!(LO, RO) # O(D^3d^2χ)
		ϵp = norm(x2) # this can be an estimation of the projection error

		# normal svd, O(D^3d^3)
		@timeit TO "svd" Al_ex, s, Ar_ex, ϵ = _tsvd_try(x2; trunc = truncdim(D_add) & truncbelow(Alg.tol))
		rmul!(Al_ex, 0.0)
	end
	info = BondInfo(s, ϵ)

	# oplus
	@timeit TO "oplus" begin
		Al_f = catdomain(Al_perm, Al_ex)
		Ar_f = catcodomain(RO.Ar_c, Ar_ex)
	end

	if Alg.check
		@timeit TO "check" ϵ = norm(Al_f * Ar_f - Al_perm * RO.Ar_c)
	else
		ϵ = NaN
	end

	return Al_f, Ar_f, CBEInfo(Alg, (info,), dim(Ar, 1), dim(Ar_f, 1), ϵp, ϵ)

end

function _orth!(LO::LeftOrthComplement)
	if get_num_workers() > 1
		@assert false "not implemented"
	elseif get_num_threads_julia() > 0 # multi-threading
		Threads.@threads :greedy for (Al, El) in zip(LO.Al, LO.El)
			# Al -= Al_c * El
			mul!(Al, LO.Al_c, El, -1.0, 1.0)
		end
	else
		for (Al, El) in zip(LO.Al, LO.El)
			# Al -= Al_c * El
			mul!(Al, LO.Al_c, El, -1.0, 1.0)
		end
	end
	return LO
end

function _orth!(RO::RightOrthComplement)
	if get_num_workers() > 1
		@assert false "not implemented"
	elseif get_num_threads_julia() > 0 # multi-threading
		Threads.@threads :greedy for (Ar, Er) in zip(RO.Ar, RO.Er)
			# Ar -= Er * Ar_c
			mul!(Ar, Er, RO.Ar_c, -1.0, 1.0)
		end
	else
		for (Ar, Er) in zip(RO.Ar, RO.Er)
			# Ar -= Er * Ar_c
			mul!(Ar, Er, RO.Ar_c, -1.0, 1.0)
		end
	end
	return RO
end

function _contractLR!(LO::LeftOrthComplement, RO::RightOrthComplement)
	# contract LO and RO, note this function will destroy LO and RO
	TT = promote_type(eltype(LO.Al_c), eltype(RO.Ar_c))
	x2 = zeros(TT, codomain(LO.Al[1]), domain(RO.Ar[1]))

	if get_num_workers() > 1
		@assert false "not implemented"
	elseif get_num_threads_julia() > 0 # multi-threading
		Lock = ReentrantLock()
		Threads.@threads :greedy for (Al, Ar) in zip(LO.Al, RO.Ar)
			# Al * Ar 
			x2_i = Al * Ar
			lock(Lock)
			try
				add!(x2, x2_i)
			catch e
				rethrow(e)
			finally
				unlock(Lock)
			end
		end
	else
		for (Al, Ar) in zip(LO.Al, RO.Ar)
			# x2 += Al * Ar 
			mul!(x2, Al, Ar, 1.0, 1.0)
		end
	end
	return x2
end

function _rsvd_trunc(V::ProductSpace{T}, D::Int64) where T <: GradedSpace
	V_trunc = fuse(V)
	ratio = D / dim(V_trunc)
	for (c, d) in V_trunc.dims
		# truncate dimensions
		V_trunc.dims[c] = ceil(Int64, d * ratio)
	end
	return V_trunc
end

function _rsvd_trunc(V::ProductSpace{T}, D::Int64) where T <: ElementarySpace
	V_trunc = fuse(V)
	ratio = D / dim(V_trunc)
	return T(ceil(Int64, V_trunc.d * ratio), V_trunc.dual)
end

function _apply_Ω(RO::RightOrthComplement{N}, Ω::AbstractTensorMap) where N

	RO_trunc = RightOrthComplement(similar(RO.Er), similar(RO.Ar), RO.Ar_c)
	if get_num_workers() > 1
		@assert false "not implemented"
	elseif get_num_threads_julia() > 0 # multi-threading
		# Ar = Ar * Ω
		Threads.@threads :greedy for i in 1:N
			RO_trunc.Ar[i] = RO.Ar[i] * Ω
		end
	else
		for i in 1:N
			RO_trunc.Ar[i] = RO.Ar[i] * Ω
		end
	end
	return RO_trunc
end

function _apply_Ω(LO::LeftOrthComplement{N}, Ω::AbstractTensorMap) where N

	LO_trunc = LeftOrthComplement(similar(LO.El), similar(LO.Al), LO.Al_c)
	if get_num_workers() > 1
		@assert false "not implemented"
	elseif get_num_threads_julia() > 0 # multi-threading
		# Al = Ω * Al
		Threads.@threads :greedy for i in 1:N
			LO_trunc.Al[i] = Ω * LO.Al[i]
		end
	else
		for i in 1:N
			LO_trunc.Al[i] = Ω * LO.Al[i]
		end
	end
	return LO_trunc
end
