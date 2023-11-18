function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor, Alg::NoCBE; kwargs...)
     Al_ex, Ar_ex, info, to = _CBE(Al, Ar, Alg; kwargs...)
     D₀ = D = dim(Ar_ex, 1)
     return Al_ex, Ar_ex, CBEInfo(Alg, info, D₀, D, 0)
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

     D₀ = dim(Ar, 1)
     D = dim(Ar_ex, 1)
     return Al_ex, Ar_ex, CBEInfo(Alg, info, D₀, D, ϵ)
end

function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::StandardCBE{T}; kwargs...) where {R₁,R₂,T<:Union{SweepL2R,SweepR2L}}

     try

          LocalTimer = reset_timer!(get_timer("CBE"))

          Dl = mapreduce(idx -> dim(Al, idx)[2], *, 1:R₁-1)
          Dr = mapreduce(idx -> dim(Ar, idx)[2], *, 2:R₂)
          Dc = dim(Ar, 1)[2]
          # if Dl ≤ Dc || Dr ≤ Dc # already full
          if Dl ≤ Dc || Dr ≤ Dc || T <: SweepL2R # only test R2L 
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

          D₀ = dim(Ar, 1)
          D = dim(Ar_ex, 1)
          return Al_ex, Ar_ex, CBEInfo(Alg, info, D₀, D, ϵ)

     catch
          jldsave("CBEerr_$(time_ns()).jld2"; PH, Al, Ar, Alg)
     end
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
          Al_lc::MPSTensor, ~, info1 = leftorth(Al; trunc=truncbelow(Alg.tol))
     end
     # left orthogonal complement, note the bond tensor is in it
     @timeit LocalTimer "ConstructLO" LO = LeftOrthComplement(PH.El, Al_lc, PH.H[1], Al)
     # 2-nd svd, implemented by eig
     @timeit LocalTimer "svd2" C::MPSTensor, info2 = _CBE_leftorth_R(LO; trunc=truncbelow(Alg.tol), normalize=true)
     # right orthogonal complement
     @timeit LocalTimer "ConstructRO" RO = RightOrthComplement(PH.Er, Ar_rc, PH.H[2])

     # 3-rd svd, D -> D/w, weighted by the updated bond tensor C
     @show w = mapreduce(x -> rank(x) == 2 ? 1 : dim(x, 2)[2], +, RO.Er)
     @timeit LocalTimer "svd3" RO_trunc, info3 = _CBE_leftorth_R(RO;
          BondTensor=C,
          trunc=truncbelow(Alg.tol) & truncdim(div(Alg.D, w)))

     # 4-th svd, preselect
     @timeit LocalTimer "preselect" Ar_fuse::MPSTensor = _preselect(RO_trunc)

     # directly apply svd
     @timeit LocalTimer "svd4" begin
          ~, Ar_pre::MPSTensor, info4 = rightorth(Ar_fuse;
               trunc=truncbelow(Alg.tol) & truncdim(Alg.D))
     end
     # final select
     @timeit LocalTimer "finalselect" begin
          Er_trunc = _initialize_Er(RO.Ar, Ar_pre)
          Al_final::MPSTensor = _finalselect(LO, Er_trunc)
     end


     # # direct sum
     # @timeit LocalTimer "expand" begin
     #      Ar_ex::MPSTensor = _directsum_Ar(Ar, Ar_final)
     #      Al_ex::MPSTensor = _expand_Al(Ar, Ar_ex, Al)
     # end

     # return Al_ex, Ar_ex, (info1, info2, info3, info4, info5), LocalTimer
end

function _CBE(PH::SparseProjectiveHamiltonian{2}, Al_lc::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::StandardCBE{SweepR2L}; kwargs...) where {R₁,R₂}

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
     w = mapreduce(x -> rank(x) == 2 ? 1 : dim(x, 2)[2], +, LO.El)
     @timeit LocalTimer "svd3" LO_trunc, info3 = _CBE_rightorth_L(LO;
          BondTensor=C,
          trunc=truncbelow(Alg.tol) & truncdim(div(Alg.D, w)))

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
          Ar_final::MPSTensor = _finalselect(El_trunc, RO)
     end
     # 5-th svd, directly use svd 
     D₀ = dim(Al_lc, R₁)[2] # original bond dimension
     @timeit LocalTimer "svd5" begin
          U::MPSTensor, ~, ~, info5 = tsvd(Ar_final, (1,), Tuple(2:R₂); trunc=truncbelow(Alg.tol) & truncdim(Alg.D - D₀))
     end
     Al_final::MPSTensor = Al_pre * U

     # direct sum
     @timeit LocalTimer "expand" begin
          Al_ex::MPSTensor = _directsum_Al(Al_lc, Al_final)
          Ar_ex::MPSTensor = _expand_Ar(Al_lc, Al_ex, Ar)
     end

     return Al_ex, Ar_ex, (info1, info2, info3, info4, info5), LocalTimer
end

