function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor, Alg::NoCBE; kwargs...)
     Al_ex, Ar_ex, info, to = _CBE(Al, Ar, Alg; kwargs...)
     return Al_ex, Ar_ex, CBEInfo(Alg, info, 0)
end

function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::FullCBE{T}; kwargs...) where {R₁,R₂,T<:Union{SweepL2R,SweepR2L}}

     LocalTimer = reset_timer!(get_timer("CBE"))
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

     if get(kwargs, :check, false)
          @timeit LocalTimer "check" ϵ = norm(Al * Ar - Al_ex * Ar_ex)
     else
          ϵ = NaN
     end

     return Al_ex, Ar_ex, CBEInfo(Alg, info, ϵ)
end

function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::StandardCBE{T}; kwargs...) where {R₁,R₂,T<:Union{SweepL2R,SweepR2L}}

     LocalTimer = reset_timer!(get_timer("CBE"))

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

     if get(kwargs, :check, false)
          @timeit LocalTimer "check" ϵ = norm(Al * Ar - Al_ex * Ar_ex)
     else
          ϵ = NaN
     end

     return Al_ex, Ar_ex, CBEInfo(Alg, info, ϵ)
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

function _CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::StandardCBE{SweepL2R}; kwargs...) where {R₁,R₂}

     LocalTimer = TimerOutput()

     # 1-st svd, bond-canonicalize Al
     @timeit LocalTimer "svd1" Al_lc::MPSTensor, S, info1 = leftorth(Al; trunc=truncbelow(Alg.tol))
     # orthogonal complement
     @timeit LocalTimer "leftnull" Al_oc = _CBE_leftnull(Al_lc)
     @timeit LocalTimer "rightnull" Ar_oc = _CBE_rightnull(Ar)

     @timeit LocalTimer "_pushright" El = _pushright(PH.El, Al_oc', PH.H[1], Al)
     # 2-nd svd, implemented by eig
     @timeit LocalTimer "svd2" C::MPSTensor, info2 = _CBE_MdM2SVd(El; trunc=truncbelow(Alg.tol), normalize=true)
     Ar_sv::MPSTensor = C * Ar

     @timeit LocalTimer "_pushleft" Er = _pushleft(PH.Er, Ar_oc', PH.H[2], Ar_sv)
     # 3-rd svd, D -> D/w
     @timeit LocalTimer "svd3" Er_trunc, info3 = _CBE_MMd2SVd(Er; trunc=truncbelow(Alg.tol) & truncdim(div(Alg.D, length(Er))), normalize=true)

     # 4-th svd
     @timeit LocalTimer "svd4" Vd::MPSTensor, info4 = _CBE_MdM2Vd(Er_trunc; trunc=truncbelow(Alg.tol) & truncdim(Alg.D), normalize=true)
     Ar_pre::MPSTensor = Vd * Ar_oc

     # final select
     @timeit LocalTimer "_pushleft" Er_final = _pushleft(PH.Er, Ar_pre', PH.H[2], Ar)
     @timeit LocalTimer "_final_contract" M::MPSTensor = _final_contract(El, Er_final)

     # 5-th svd, directly use svd
     D₀ = dim(Ar, 1)[2] # original bond dimension
     @timeit LocalTimer "svd5" begin
          ~, Q::MPSTensor, info5 = rightorth(M; trunc=truncbelow(Alg.tol) & truncdim(Alg.D - D₀))
     end
     Ar_final::MPSTensor = Q * Ar_pre

     # direct sum
     @timeit LocalTimer "expand" begin
          Ar_ex::MPSTensor = _directsum_Ar(Ar, Ar_final)
          Al_ex::MPSTensor = _expand_Al(Ar, Ar_ex, Al)
     end

     return Al_ex, Ar_ex, (info1, info2, info3, info4, info5), LocalTimer
end

function _CBE(PH::SparseProjectiveHamiltonian{2}, Al_lc::MPSTensor{R}, Ar::MPSTensor, Alg::StandardCBE{SweepR2L}; kwargs...) where {R}

     LocalTimer = TimerOutput()

     # 1-st svd, bond-canonicalize Ar
     @timeit LocalTimer "svd1" begin
          ~, Ar_rc::MPSTensor, info1 = rightorth(Ar; trunc=truncbelow(Alg.tol))
     end
     # right orthogonal complement, note the bond tensor is in it
     @timeit LocalTimer "ConstructRO" RO = RightOrthComplement(PH.Er, Ar_rc, PH.H[2], Ar)
     # 2-nd svd, implemented by eig
     @timeit LocalTimer "svd2" C::MPSTensor, info2 = _CBE_rightorth_L(RO; trunc=truncbelow(Alg.tol), normalize=true)
     # left orthogonal complement
     @timeit LocalTimer "ConstructLO" LO = LeftOrthComplement(PH.El, Al_lc, PH.H[1])

     # 3-rd svd, D -> D/w, weighted by the updated bond tensor C
     @timeit LocalTimer "svd3" LO_trunc, info3 = _CBE_rightorth_L(LO;
          BondTensor=C,
          trunc=truncbelow(Alg.tol) & truncdim(div(Alg.D, length(LO))))

     # 4-th svd, preselect
     @timeit LocalTimer "preselect" Al_fuse::MPSTensor = _preselect(LO_trunc)
     # directly apply svd
     @timeit LocalTimer "svd4" begin
          Al_pre::MPSTensor, ~, info4 = leftorth(Al_fuse;
               trunc=truncbelow(Alg.tol) & truncdim(Alg.D))
     end
     # final select
     @timeit LocalTimer "finalselect" begin
          El_trunc = _initialize_El(LO.Al, Al_pre)
          # M::MPSTensor = _final_contract(El_final, Er)
     end
     # # 5-th svd, directly use svd 
     # D₀ = dim(Ar, 1)[2] # original bond dimension
     # @timeit LocalTimer "svd5" begin
     #      Q::MPSTensor, ~, info5 = leftorth(M; trunc=truncbelow(Alg.tol) & truncdim(Alg.D - D₀))
     # end
     # Al_final::MPSTensor = Al_pre * Q

     # # direct sum
     # @timeit LocalTimer "expand" begin
     #      Al_ex::MPSTensor = _directsum_Al(Al, Al_final)
     #      Ar_ex::MPSTensor = _expand_Ar(Al, Al_ex, Ar)
     # end

     # return Al_ex, Ar_ex, (info1, info2, info3, info4, info5), LocalTimer
end

