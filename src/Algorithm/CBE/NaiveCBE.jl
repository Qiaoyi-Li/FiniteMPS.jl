function _CBE(Al::MPSTensor{R1}, Ar::MPSTensor{R2}, El::SparseLeftTensor, Er::SparseRightTensor, Hl::SparseMPOTensor, Hr::SparseMPOTensor, Alg::NaiveCBE{SweepR2L}, TO::TimerOutput) where {R1, R2}
	# Al is left canonical, Ar is canonical center

	# right canonical Ar 
	@timeit TO "permute" Ar_perm = permute(Ar.A, ((1,), Tuple(2:R2)))
	@timeit TO "qr" _, Ar_c::MPSTensor = rightorth(Ar_perm)

	# construct L/R orth complement obj 
	@timeit TO "construct LO" LO = LeftOrthComplement(El, Al, Hl)
	@timeit TO "construct RO" RO = RightOrthComplement(Er, Ar_c, Hr, Ar)

	# contract LO and RO
	@timeit TO "LO * RO" x2 = _contractLR!(LO, RO)
	ϵp = norm(x2) # this can be an estimation of the projection error

	# svd, note x2 will be destroyed
	D_add = Alg.D - dim(Al, R2)[2]
	@timeit TO "svd" Al_ex, s, vd, ϵ = tsvd!(x2; trunc = truncdim(D_add) & truncbelow(Alg.tol))
	info = BondInfo(s, ϵ)

	# oplus
	@timeit TO "oplus" begin
		Al_f = catdomain(LO.Al_c, Al_ex)
		Ar_f = catcodomain(Ar_perm, rmul!(vd, 0.0))
	end

	if Alg.check
		@timeit TO "check" ϵ = norm(Al_f * Ar_f - LO.Al_c * Ar_perm)
	else
		ϵ = NaN
	end

	return Al_f, Ar_f, CBEInfo(Alg, (info,), dim(Ar, 1), dim(Ar_f, 1), ϵp, ϵ)
end

function _CBE(Al::MPSTensor{R1}, Ar::MPSTensor{R2}, El::SparseLeftTensor, Er::SparseRightTensor, Hl::SparseMPOTensor, Hr::SparseMPOTensor, Alg::NaiveCBE{SweepL2R}, TO::TimerOutput) where {R1, R2}
	# Al is canonical center, Ar is right canonical

	# left canonical Al
	@timeit TO "permute" Al_perm = permute(Al.A, (Tuple(1:R1-1), (R1,)))
	@timeit TO "qr" Al_c::MPSTensor, _ = leftorth(Al_perm)

	# construct L/R orth complement obj
	@timeit TO "construct LO" LO = LeftOrthComplement(El, Al_c, Hl, Al)
	@timeit TO "construct RO" RO = RightOrthComplement(Er, Ar, Hr)

	# contract LO and RO
	@timeit TO "LO * RO" x2 = _contractLR!(LO, RO)
	ϵp = norm(x2) # this can be an estimation of the projection error

	# svd, note x2 will be destroyed
	D_add = Alg.D - dim(Ar, 1)[2]
	@timeit TO "svd" u, s, Ar_ex, ϵ = tsvd!(x2; trunc = truncdim(D_add) & truncbelow(Alg.tol))
	info = BondInfo(s, ϵ)

	# oplus
     @timeit TO "oplus" begin
	Al_f = catdomain(Al_perm, rmul!(u, 0.0))
	Ar_f = catcodomain(RO.Ar_c, Ar_ex)
     end

	if Alg.check
		@timeit TO "check" ϵ = norm(Al_f * Ar_f - Al_perm * RO.Ar_c)
	else
		ϵ = NaN
	end

	return Al_f, Ar_f, CBEInfo(Alg, (info,), dim(Ar, 1), dim(Ar_f, 1), ϵp, ϵ)

end


function _contractLR!(LO::LeftOrthComplement, RO::RightOrthComplement)
	# contract LO and RO, note this function will destroy LO and RO
	TT = promote_type(eltype(LO.Al_c), eltype(RO.Ar_c))
	x2 = zeros(TT, codomain(LO.Al_c), domain(RO.Ar_c))

	if get_num_workers() > 1
		@assert false "not implemented"
	elseif get_num_threads_julia() > 0 # multi-threading
		Threads.@threads :greedy for (Al, El) in zip(LO.Al, LO.El)
			# Al -= Al_c * El
			mul!(Al, LO.Al_c, El, -1.0, 1.0)
		end
		Threads.@threads :greedy for (Ar, Er) in zip(RO.Ar, RO.Er)
			# Ar -= Er * Ar_c
			mul!(Ar, Er, RO.Ar_c, -1.0, 1.0)
		end
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
		for (Al, El, Ar, Er) in zip(LO.Al, LO.El, RO.Ar, RO.Er)
			# Al -= Al_c * El
			mul!(Al, LO.Al_c, El, -1.0, 1.0)
			# Ar -= Er * Ar_c
			mul!(Ar, Er, RO.Ar_c, -1.0, 1.0)
			# x2 += Al * Ar 
			mul!(x2, Al, Ar, 1.0, 1.0)
		end
	end
	return x2
end
