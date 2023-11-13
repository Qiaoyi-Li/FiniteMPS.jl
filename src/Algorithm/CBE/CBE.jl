"""
     abstract type CBEDirection end

Direction of CBE, wrapped as a type to be passed as a type parameter. 
"""
abstract type CBEDirection end

"""
     struct L2RCBE <: CBEDirection end
"""
struct L2RCBE <: CBEDirection end

"""
     struct R2LCBE <: CBEDirection end
"""
struct R2LCBE <: CBEDirection end

""" 
     abstract type CBEAlgorithm{T <: CBEDirection}

Abstract type of all (controlled bond expansion) CBE algorithms.
"""
abstract type CBEAlgorithm{T<:CBEDirection} end

"""
     struct StandardCBE{T} <: CBEAlgorithm{T} where T <: CBEDirection
          D::Int64
          tol::Int64
     end

Standard CBE algorithm, details see [PhysRevLett.130.246402](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.246402).
"""
struct StandardCBE{T <: CBEDirection} <: CBEAlgorithm{T}
     D::Int64
     tol::Float64
     function StandardCBE(Direction::Symbol, D::Int64, tol::Float64)
          @assert Direction ∈ (:L, :R)
          @assert D > 0
          @assert tol ≥ 0
          if Direction == :L
               return new{L2RCBE}(D, tol)
          else
               return new{R2LCBE}(D, tol)
          end
     end
end


function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor, Alg::StandardCBE{L2RCBE}; kwargs...)

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

function CBE(PH::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor, Alg::StandardCBE{R2LCBE}; kwargs...)

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

