function _CBE(Al::MPSTensor, Ar::MPSTensor, Alg::FullCBE{SweepL2R}, TO::TimerOutput)

     @timeit TO "contract" x2 = CompositeMPSTensor(Al, Ar)
	@timeit TO "qr" Al_f, Ar_f, info = rightorth(x2; trunc = notrunc())

     if Alg.check
		@timeit TO "check" ϵ = norm(Al_f * Ar_f - x2.A)
	else
		ϵ = NaN 
	end

	return Al_f, Ar_f, CBEInfo(Alg, (info,), dim(Ar, 1), dim(Ar_f, 1), NaN, ϵ)
end

function _CBE(Al::MPSTensor, Ar::MPSTensor, Alg::FullCBE{SweepR2L}, TO::TimerOutput)

     @timeit TO "contract" x2 = CompositeMPSTensor(Al, Ar)
	@timeit TO "qr" Al_f, Ar_f, info = leftorth(x2; trunc = notrunc())

     if Alg.check
		@timeit TO "check" ϵ = norm(Al_f * Ar_f - x2.A)
	else
		ϵ = NaN 
	end

	return Al_f, Ar_f, CBEInfo(Alg, (info,), dim(Ar, 1), dim(Ar_f, 1), NaN, ϵ)
end