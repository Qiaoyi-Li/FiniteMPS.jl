# TODO add Timers
function CBE(Al::MPSTensor, Ar::MPSTensor, Alg::NoCBE; kwargs...)
     return Al, Ar, nothing
end

function CBE(Al::MPSTensor, Ar::MPSTensor, Alg::FullCBE{SweepL2R}; kwargs...)


     if get(kwargs, :reverse, false)
          Al_ex, tmp, ~ = CBE(Al, Ar, FullCBE(SweepR2L()))
          # canonicalize without truncation  
          S::MPSTensor, Ar_ex::MPSTensor, info = rightorth(tmp; trunc=notrunc())
          return convert(MPSTensor, Al_ex * S), Ar_ex, info
     else
          Ar_ex = _isometry_Ar(Ar)
          Al_ex = _expand_Al(Ar, Ar_ex, Al)
          return Al_ex, Ar_ex, nothing
     end
end

function CBE(Al::MPSTensor, Ar::MPSTensor, Alg::FullCBE{SweepR2L}; kwargs...)

     if get(kwargs, :reverse, false)
          tmp, Ar_ex, ~ = CBE(Al, Ar, FullCBE(SweepL2R()))
          # canonicalize without truncation  
          Al_ex::MPSTensor, S::MPSTensor, info = leftorth(tmp; trunc=notrunc())
          return Al_ex, convert(MPSTensor, S * Ar_ex), info
     else
          Al_ex = _isometry_Al(Al)
          Ar_ex = _expand_Ar(Al, Al_ex, Ar)
          return Al_ex, Ar_ex, nothing
     end
end

function CBE(::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor, Alg::T; kwargs...) where {T<:Union{NoCBE,FullCBE}}
     return CBE(Al, Ar, Alg; kwargs...)
end

function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::StandardCBE{SweepL2R}; kwargs...) where {R₁,R₂}

     Dl = mapreduce(idx -> dim(Al, idx)[2], *, 1:R₁-1)
     Dr = mapreduce(idx -> dim(Ar, idx)[2], *, 2:R₂)
     if Dl == dim(Al, R₁)[2] # already full
          return CBE(Al, Ar, NoCBE(SweepL2R()))
     elseif Dr ≤ Alg.D # full cbe is not expensive
          return CBE(Al, Ar, FullCBE(SweepL2R()))
     elseif Dl < Alg.D
          return CBE(Al, Ar, FullCBE(SweepL2R()); reverse=true)
     end


     info = Vector{BondInfo}(undef, 4) # truncate info of 4 times svd

     # bond-canonicalize Al
     Al_lc::MPSTensor, S = leftorth(Al; trunc=truncbelow(Alg.tol))
     # orthogonal complement
     Al_oc = _CBE_leftnull(Al_lc)
     Ar_oc = _CBE_rightnull(Ar)

     El = _pushright(PH.El, Al_oc', PH.H[1], Al)
     # 1-st svd, implemented by eig
     C::MPSTensor, info[1] = _CBE_MdM2SVd(El; trunc=truncbelow(Alg.tol), normalize=true)
     Ar_sv::MPSTensor = C * Ar

     Er = _pushleft(PH.Er, Ar_oc', PH.H[2], Ar_sv)
     # 2-nd svd, D -> D/w
     Er_trunc, info[2] = _CBE_MMd2SVd(Er; trunc=truncbelow(Alg.tol) & truncdim(div(Alg.D, length(Er))), normalize=true)

     # 3-rd svd
     Vd::MPSTensor, info[3] = _CBE_MdM2Vd(Er_trunc; trunc=truncbelow(Alg.tol) & truncdim(Alg.D), normalize=true)
     Ar_pre::MPSTensor = Vd * Ar_oc

     # final select
     Er_final = _pushleft(PH.Er, Ar_pre', PH.H[2], Ar)
     M::MPSTensor = _final_contract(El, Er_final)

     # 4-th svd, directly use svd
     D₀ = dim(Ar, 1)[2] # original bond dimension
     ~, Q::MPSTensor, info[4] = rightorth(M; trunc=truncbelow(Alg.tol) & truncdim(Alg.D - D₀))
     Ar_final::MPSTensor = Q * Ar_pre

     # direct sum
     Ar_ex::MPSTensor = _directsum_Ar(Ar, Ar_final)
     Al_ex::MPSTensor = _expand_Al(Ar, Ar_ex, Al)

     return Al_ex, Ar_ex, info
end

function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor{R₁}, Ar::MPSTensor{R₂}, Alg::StandardCBE{SweepR2L}; kwargs...) where {R₁,R₂}

     Dl = mapreduce(idx -> dim(Al, idx)[2], *, 1:R₁-1)
     Dr = mapreduce(idx -> dim(Ar, idx)[2], *, 2:R₂)
     if Dr == dim(Al, 1)[2] # already full
          return CBE(Al, Ar, NoCBE(SweepR2L()))
     elseif Dl ≤ Alg.D # full cbe is not expensive
          return CBE(Al, Ar, FullCBE(SweepR2L()))
     elseif Dr < Alg.D
          return CBE(Al, Ar, FullCBE(SweepR2L()); reverse=true)
     end

     info = Vector{BondInfo}(undef, 4) # truncate info of 4 times svd

     # bond-canonicalize Ar
     S, Ar_rc::MPSTensor = rightorth(Ar; trunc=truncbelow(Alg.tol))
     # orthogonal complement
     Al_oc = _CBE_leftnull(Al)
     Ar_oc = _CBE_rightnull(Ar_rc)

     Er = _pushleft(PH.Er, Ar_oc', PH.H[2], Ar)
     # 1-st svd, implemented by eig
     C::MPSTensor, info[1] = _CBE_MMd2US(Er; trunc=truncbelow(Alg.tol), normalize=true)
     Al_us::MPSTensor = Al * C

     El = _pushright(PH.El, Al_oc', PH.H[1], Al_us)
     # 2-nd svd, D -> D/w
     El_trunc, info[2] = _CBE_MdM2US(El; trunc=truncbelow(Alg.tol) & truncdim(div(Alg.D, length(El))), normalize=true)

     # 3-rd svd
     U::MPSTensor, info[3] = _CBE_MMd2U(El_trunc; trunc=truncbelow(Alg.tol) & truncdim(Alg.D), normalize=true)
     Al_pre::MPSTensor = Al_oc * U

     # final select
     El_final = _pushright(PH.El, Al_pre', PH.H[1], Al)
     M::MPSTensor = _final_contract(El_final, Er)

     # 4-th svd, directly use svd 
     D₀ = dim(Ar, 1)[2] # original bond dimension
     Q::MPSTensor, ~, info[4] = leftorth(M; trunc=truncbelow(Alg.tol) & truncdim(Alg.D - D₀))
     Al_final::MPSTensor = Al_pre * Q

     # direct sum
     Al_ex::MPSTensor = _directsum_Al(Al, Al_final)
     Ar_ex::MPSTensor = _expand_Ar(Al, Al_ex, Ar)

     return Al_ex, Ar_ex, info
end

