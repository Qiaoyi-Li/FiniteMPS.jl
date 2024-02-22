"""
     CBE(PH::SparseProjectiveHamiltonian{2},
          Al::MPSTensor, Ar::MPSTensor,
          Alg::CBEAlgorithm;
          kwargs...) -> Al_ex::MPSTensor, Ar_ex::MPSTensor, info::CBEInfo

Return the two local tensors `Al_ex` and `Ar_ex` after CBE.
"""
function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor, Alg::NoCBE; kwargs...)
     Al_ex, Ar_ex, info, _ = _CBE(Al, Ar, Alg; kwargs...)
     D₀ = D = dim(Ar_ex, 1)
     return Al_ex, Ar_ex, CBEInfo(Alg, info, D₀, D, 0)
end

function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::FullCBE{T}; kwargs...) where {R₁,R₂,T<:Union{SweepL2R,SweepR2L}}

     LocalTimer = reset_timer!(get_timer("CBE"))
     cbecheck::Bool = Alg.check
     Dl = mapreduce(idx -> dim(Al, idx)[2], *, 1:R₁-1)
     Dr = mapreduce(idx -> dim(Ar, idx)[2], *, 2:R₂)
     Dc = dim(Ar, 1)[2]
     if Dl ≤ Dc || Dr ≤ Dc # already full
          Alg = NoCBE(T())
          Al_ex, Ar_ex, info, to = _CBE(Al, Ar, Alg; kwargs...)
     else
          @timeit LocalTimer "FullCBE" Al_ex, Ar_ex, info, to = _CBE(Al, Ar, Alg; kwargs...)
          merge!(LocalTimer, to; tree_point=["FullCBE"])
     end

     if cbecheck
          @timeit LocalTimer "check" ϵ = norm(Al * Ar - Al_ex * Ar_ex)
     else
          ϵ = NaN
     end

     D₀ = dim(Ar, 1)
     D = dim(Ar_ex, 1)
     return Al_ex, Ar_ex, CBEInfo(Alg, info, D₀, D, ϵ)
end

function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::StandardCBE{T}; kwargs...) where {R₁,R₂,T<:Union{SweepL2R,SweepR2L}}

     LocalTimer = reset_timer!(get_timer("CBE"))
     cbecheck::Bool = Alg.check
     Dl = mapreduce(idx -> dim(Al, idx)[2], *, 1:R₁-1)
     Dr = mapreduce(idx -> dim(Ar, idx)[2], *, 2:R₂)
     Dc = dim(Ar, 1)[2]
     if Dl ≤ Dc || Dr ≤ Dc # already full
          Alg = NoCBE(T())
          Al_ex, Ar_ex, info, to = _CBE(Al, Ar, Alg; kwargs...)
     elseif Dl ≤ Alg.D || Dr ≤ Alg.D # full cbe is not expensive
          Alg = FullCBE(T())
          @timeit LocalTimer "FullCBE" Al_ex, Ar_ex, info, to = _CBE(Al, Ar, Alg; kwargs...)
          merge!(LocalTimer, to; tree_point=["FullCBE"])
     else
          @timeit LocalTimer "StandardCBE" Al_ex, Ar_ex, info, to = _CBE(PH, Al, Ar, Alg; kwargs...)
          merge!(LocalTimer, to; tree_point=["StandardCBE"])
     end

     if cbecheck
          @timeit LocalTimer "check" ϵ = norm(Al * Ar - Al_ex * Ar_ex)
     else
          ϵ = NaN
     end

     D₀ = dim(Ar, 1)
     D = dim(Ar_ex, 1)
     return Al_ex, Ar_ex, CBEInfo(Alg, info, D₀, D, ϵ)

end

function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::CheapCBE{T}; kwargs...) where {R₁,R₂,T<:Union{SweepL2R,SweepR2L}}

     LocalTimer = reset_timer!(get_timer("CBE"))
     cbecheck::Bool = Alg.check
     Dl = mapreduce(idx -> dim(Al, idx)[2], *, 1:R₁-1)
     Dr = mapreduce(idx -> dim(Ar, idx)[2], *, 2:R₂)
     Dc = dim(Ar, 1)[2]
     if Dl ≤ Dc || Dr ≤ Dc # already full
          Alg = NoCBE(T())
          Al_ex, Ar_ex, info, to = _CBE(Al, Ar, Alg; kwargs...)
     elseif Dl ≤ Alg.D || Dr ≤ Alg.D # full cbe is not expensive
          Alg = FullCBE(T())
          @timeit LocalTimer "FullCBE" Al_ex, Ar_ex, info, to = _CBE(Al, Ar, Alg; kwargs...)
          merge!(LocalTimer, to; tree_point=["FullCBE"])
     else
          @timeit LocalTimer "CheapCBE" Al_ex, Ar_ex, info, to = _CBE(PH, Al, Ar, Alg; kwargs...)
          merge!(LocalTimer, to; tree_point=["CheapCBE"])
     end

     if cbecheck
          @timeit LocalTimer "check" ϵ = norm(Al * Ar - Al_ex * Ar_ex)
     else
          ϵ = NaN
     end

     D₀ = dim(Ar, 1)
     D = dim(Ar_ex, 1)
     return Al_ex, Ar_ex, CBEInfo(Alg, info, D₀, D, ϵ)

end

# ================== implementation ==================
# return Al, Ar, info, LocalTimer
function _CBE(Al::MPSTensor, Ar::MPSTensor, Alg::NoCBE; kwargs...)
     return Al, Ar, (), TimerOutput()
end

function _CBE(Al::MPSTensor, Ar::MPSTensor, Alg::FullCBE{SweepL2R}; kwargs...)

     LocalTimer = TimerOutput()
     @timeit LocalTimer "rightorth" Al_ex::MPSTensor, Ar_ex::MPSTensor, info = rightorth(CompositeMPSTensor(Al, Ar); trunc=notrunc())

     return Al_ex, Ar_ex, (info,), LocalTimer
end

function _CBE(Al::MPSTensor, Ar::MPSTensor, Alg::FullCBE{SweepR2L}; kwargs...)

     LocalTimer = TimerOutput()
     @timeit LocalTimer "leftorth" Al_ex::MPSTensor, Ar_ex::MPSTensor, info = leftorth(CompositeMPSTensor(Al, Ar); trunc=notrunc())

     return Al_ex, Ar_ex, (info,), LocalTimer
end

function _CBE(::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor, Alg::T; kwargs...) where {T<:Union{NoCBE,FullCBE}}
     return _CBE(Al, Ar, Alg; kwargs...)
end

function _CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar_rc::MPSTensor{R₂}, Alg::StandardCBE{SweepL2R}; kwargs...) where {R₁,R₂}

     LocalTimer = TimerOutput()

     # 1-st svd, bond-canonicalize Al
     @timeit LocalTimer "svd1" begin
          Al_lc::MPSTensor, _, info1 = leftorth(Al; trunc=truncbelow(Alg.tol))
     end
     # left orthogonal complement, note the bond tensor is in it
     @timeit LocalTimer "ConstructLO" LO = LeftOrthComplement(PH.El, Al_lc, PH.H[1], Al)
     # 2-nd svd, implemented by eig
     @timeit LocalTimer "svd2" C::MPSTensor, info2 = _CBE_leftorth_R(LO; trunc=truncbelow(Alg.tol), normalize=true)
     # right orthogonal complement
     @timeit LocalTimer "ConstructRO" RO = RightOrthComplement(PH.Er, Ar_rc, PH.H[2])

     # 3-rd svd, D -> D/w, weighted by the updated bond tensor C
     w = mapreduce(x -> rank(x) == 2 ? 1 : dim(x, 2)[2], +, RO.Er)
     @timeit LocalTimer "svd3" RO_trunc, info3 = _CBE_leftorth_R(RO;
          BondTensor=C,
          trunc=truncbelow(Alg.tol) & truncdim(div(Alg.D, w)))

     # 4-th svd, preselect
     @timeit LocalTimer "preselect" Ar_fuse::MPSTensor = _preselect(RO_trunc)

     # directly apply svd
     @timeit LocalTimer "svd4" begin
          _, Ar_pre::MPSTensor, info4 = rightorth(Ar_fuse;
               trunc=truncbelow(Alg.tol) & truncdim(Alg.D))
     end
     # final select
     @timeit LocalTimer "finalselect" begin
          Er_trunc = _initialize_Er(RO.Ar, Ar_pre)
          Al_final::MPSTensor = _finalselect(LO, Er_trunc)
          # @show norm(permute(Al_final.A, (1, 2, 3), (4,))' * permute(LO.Al_c.A, (1, 2, 3), (4,)))
     end

     # 5-th svd, directly use svd
     D₀ = dim(Ar_rc, 1)[2] # original bond dimension
     @timeit LocalTimer "svd5" begin
          _, _, Vd::MPSTensor, info5 = tsvd(Al_final, Tuple(1:R₁-1), (R₁,); trunc=truncbelow(Alg.tol) & truncdim(Alg.D - D₀))
     end
     Ar_final::MPSTensor = Vd * Ar_pre
     # orthogonalize again
     @timeit LocalTimer "reortho" begin
          axpy!(-1, _rightProj(Ar_final, Ar_rc), Ar_final.A)
     end

     # direct sum
     @timeit LocalTimer "oplus" Ar_ex::MPSTensor = _directsum_Ar(Ar_rc, Ar_final)
     @timeit LocalTimer "expand" Al_ex::MPSTensor = _expand_Al(Ar_rc, Ar_ex, Al)

     @timeit LocalTimer "svd6" begin
          S::MPSTensor, Ar_ex, info6 = rightorth(Ar_ex; trunc=notrunc())
          Al_ex = Al_ex * S
     end

     return Al_ex, Ar_ex, (info1, info2, info3, info4, info5, info6), LocalTimer
end

function _CBE(PH::SparseProjectiveHamiltonian{2}, Al_lc::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::StandardCBE{SweepR2L}; kwargs...) where {R₁,R₂}

     LocalTimer = TimerOutput()

     # 1-st svd, bond-canonicalize Ar
     @timeit LocalTimer "svd1" begin
          _, Ar_rc::MPSTensor, info1 = rightorth(Ar; trunc=truncbelow(Alg.tol))
     end
     # right orthogonal complement, note the bond tensor is in it
     @timeit LocalTimer "ConstructRO" RO = RightOrthComplement(PH.Er, Ar_rc, PH.H[2], Ar)
     # 2-nd svd, implemented by eig
     @timeit LocalTimer "svd2" C::MPSTensor, info2 = _CBE_rightorth_L(RO; trunc=truncbelow(Alg.tol), normalize=true)
     # left orthogonal complement
     @timeit LocalTimer "ConstructLO" LO = LeftOrthComplement(PH.El, Al_lc, PH.H[1])

     # 3-rd svd, D -> D/w, weighted by the updated bond tensor C
     w = mapreduce(x -> rank(x) == 2 ? 1 : dim(x, 2)[2], +, LO.El)
     @timeit LocalTimer "svd3" LO_trunc, info3 = _CBE_rightorth_L(LO;
          BondTensor=C,
          trunc=truncbelow(Alg.tol) & truncdim(div(Alg.D, w)))

     # 4-th svd, preselect
     @timeit LocalTimer "preselect" Al_fuse::MPSTensor = _preselect(LO_trunc)
     # directly apply svd
     @timeit LocalTimer "svd4" begin
          Al_pre::MPSTensor, _, info4 = leftorth(Al_fuse;
               trunc=truncbelow(Alg.tol) & truncdim(Alg.D))
          # @show norm(permute(Al_pre.A, (1, 2, 3), (4,))' * permute(Al_lc.A, (1, 2, 3), (4,)))
     end
     # final select
     @timeit LocalTimer "finalselect" begin
          El_trunc = _initialize_El(LO.Al, Al_pre)
          Ar_final::MPSTensor = _finalselect(El_trunc, RO)
          # @show norm(permute(Ar_final.A, (1,), (2, 3, 4)) * permute(RO.Ar_c.A, (1,), (2, 3, 4))')
     end
     # 5-th svd, directly use svd 
     D₀ = dim(Al_lc, R₁)[2] # original bond dimension
     @timeit LocalTimer "svd5" begin
          U::MPSTensor, _, _, info5 = tsvd(Ar_final, (1,), Tuple(2:R₂); trunc=truncbelow(Alg.tol) & truncdim(Alg.D - D₀))
     end
     Al_final::MPSTensor = Al_pre * U
     # orthogonalize again
     @timeit LocalTimer "reortho" begin
          axpy!(-1, _leftProj(Al_final, Al_lc), Al_final.A)
     end
     # @show norm(permute(Al_final.A, (1, 2, 3), (4,))' * permute(Al_lc.A, (1, 2, 3), (4,)))

     # direct sum
     @timeit LocalTimer "oplus" Al_ex::MPSTensor = _directsum_Al(Al_lc, Al_final)
     @timeit LocalTimer "expand" Ar_ex::MPSTensor = _expand_Ar(Al_lc, Al_ex, Ar)


     @timeit LocalTimer "svd6" begin
          Al_ex, S::MPSTensor, info6 = leftorth(Al_ex; trunc=notrunc())
          Ar_ex = S * Ar_ex
     end

     return Al_ex, Ar_ex, (info1, info2, info3, info4, info5, info6), LocalTimer
end

function _CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar_rc::MPSTensor{R₂}, Alg::CheapCBE{SweepL2R}; kwargs...) where {R₁,R₂}

     LocalTimer = TimerOutput()

     # pre truncate
     w = mapreduce(x -> rank(x) == 2 ? 1 : dim(x, 2)[2], +, PH.Er)
     @timeit LocalTimer "svd1" begin
          _, C::MPSTensor, info1 = leftorth(Al; trunc=truncbelow(Alg.tol) & truncdim(div(Alg.D,w)))
     end
     Ar::MPSTensor = C * Ar_rc

     # right orthogonal complement
     @timeit LocalTimer "ConstructRO" RO = RightOrthComplement(PH.Er, Ar_rc, PH.H[2], Ar)

     # orthogonalize
     @timeit LocalTimer "fuseAbstractBond" Ar_fuse::MPSTensor = _preselect(RO)

     # svd to get the isometry to selected space
     D₀ = dim(codomain(Ar_rc)[1])
     @timeit LocalTimer "svd2" begin
          _, Ar_pre::MPSTensor, info2 = rightorth(Ar_fuse;
               trunc=truncbelow(Alg.tol) & truncdim(Alg.D - D₀))
     end
 
     # orthogonalize again
     @timeit LocalTimer "reortho" begin
          axpy!(-1, _rightProj(Ar_pre, Ar_rc), Ar_pre.A)
     end

     # direct sum
     @timeit LocalTimer "oplus" Ar_ex::MPSTensor = _directsum_Ar(Ar_rc, Ar_pre)
     @timeit LocalTimer "expand" Al_ex::MPSTensor = _expand_Al(Ar_rc, Ar_ex, Al)

     return Al_ex, Ar_ex, (info1, info2), LocalTimer
end

function _CBE(PH::SparseProjectiveHamiltonian{2}, Al_lc::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::CheapCBE{SweepR2L}; kwargs...) where {R₁,R₂}

     LocalTimer = TimerOutput()

     # pre truncate
     w = mapreduce(x -> rank(x) == 2 ? 1 : dim(x, 2)[2], +, PH.El)
     @timeit LocalTimer "svd1" begin
          C::MPSTensor, _, info1 = rightorth(Ar; trunc=truncbelow(Alg.tol) & truncdim(div(Alg.D,w)))
     end
     Al::MPSTensor = Al_lc * C

     # right orthogonal complement
     @timeit LocalTimer "ConstructLO" LO = LeftOrthComplement(PH.El, Al_lc, PH.H[1], Al)

     # orthogonalize
     @timeit LocalTimer "fuseAbstractBond" Al_fuse::MPSTensor = _preselect(LO)

     # svd to get the isometry to selected space
     D₀ = dim(codomain(Ar)[1])
     @timeit LocalTimer "svd2" begin
          Al_pre::MPSTensor, _, info2 = leftorth(Al_fuse;
               trunc=truncbelow(Alg.tol) & truncdim(Alg.D - D₀))
     end
 
     # orthogonalize again
     @timeit LocalTimer "reortho" begin
          axpy!(-1, _leftProj(Al_pre, Al_lc), Al_pre.A)
     end

     # direct sum
     @timeit LocalTimer "oplus" Al_ex::MPSTensor = _directsum_Al(Al_lc, Al_pre)
     @timeit LocalTimer "expand" Ar_ex::MPSTensor = _expand_Ar(Al_lc, Al_ex, Ar)

     return Al_ex, Ar_ex, (info1, info2), LocalTimer
end
