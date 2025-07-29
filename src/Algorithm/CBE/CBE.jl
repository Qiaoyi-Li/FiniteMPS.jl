"""
	 CBE(Al::MPSTensor, Ar::MPSTensor,
		  El::SparseLeftTensor, Er::SparseRightTensor,
		  Hl::SparseMPOTensor, Hr::SparseMPOTensor,
		  Alg::CBEAlgorithm
		  ) -> Al_ex::MPSTensor, Ar_ex::MPSTensor, info::CBEInfo, TO::TimerOutput

Return the two expanded local tensors `Al_ex` and `Ar_ex` after CBE.
"""
function CBE(Al::MPSTensor, Ar::MPSTensor,
	::SparseLeftTensor, ::SparseRightTensor,
	::SparseMPOTensor, ::SparseMPOTensor,
	Alg::NoCBE; kwargs...)
	D₀ = D = dim(Ar, 1)
	return Al, Ar, CBEInfo(Alg, (), D₀, D, NaN, 0.0), TimerOutput()
end

function CBE(Al::MPSTensor{R₁}, Ar::MPSTensor{R₂},
	El::SparseLeftTensor, Er::SparseRightTensor,
	Hl::SparseMPOTensor, Hr::SparseMPOTensor,
	Alg::NaiveCBE{S}; kwargs...) where {R₁, R₂, S <: Union{SweepL2R, SweepR2L}}

     
     Dl = mapreduce(idx -> dim(Al, idx)[2], *, 1:R₁-1)
	Dr = mapreduce(idx -> dim(Ar, idx)[2], *, 2:R₂)
	# use FullCBE if the full bond dimension is even smaller than Alg.D
     if Dl ≤ Alg.D || Dr ≤ Alg.D
          return CBE(Al, Ar, El, Er, Hl, Hr, FullCBE(S(); check = Alg.check))
     end

	TO = TimerOutput()
	@timeit TO "NaiveCBE" Al_ex::MPSTensor, Ar_ex::MPSTensor, info = _CBE(Al, Ar, El, Er, Hl, Hr, Alg, TO; kwargs...)

	return Al_ex, Ar_ex, info, TO
end

function CBE(Al::MPSTensor{R₁}, Ar::MPSTensor{R₂},
	El::SparseLeftTensor, Er::SparseRightTensor,
	Hl::SparseMPOTensor, Hr::SparseMPOTensor,
	Alg::FullCBE{S}) where {R₁, R₂, S <: Union{SweepL2R, SweepR2L}}

	Dl = mapreduce(idx -> dim(Al, idx)[2], *, 1:R₁-1)
	Dr = mapreduce(idx -> dim(Ar, idx)[2], *, 2:R₂)
	Dc = dim(Ar, 1)[2]
	if Dl ≤ Dc || Dr ≤ Dc # already full
		return CBE(Al, Ar, El, Er, Hl, Hr, NoCBE(S()))
	end

     TO = TimerOutput()
	@timeit TO "FullCBE" Al_ex::MPSTensor, Ar_ex::MPSTensor, info = _CBE(Al, Ar, Alg, TO)

	return Al_ex, Ar_ex, info, TO
end
