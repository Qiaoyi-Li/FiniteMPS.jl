""" 
     abstract type CBEAlgorithm{T <: SweepDirection}

Abstract type of all (controlled bond expansion) CBE algorithms.
"""
abstract type CBEAlgorithm{T<:SweepDirection} end

"""
     struct NoCBE{T <: SweepDirection} <: CBEAlgorithm{T}
"""
struct NoCBE{T <: SweepDirection} <: CBEAlgorithm{T} 
     function NoCBE(direction::T = AnyDirection()) where T <: SweepDirection
          return new{T}()
     end
end

"""
     struct FullCBE{T <: SweepDirection} <: CBEAlgorithm{T} 

Special case of CBE algorithm, directly keep the full bond space, usually used near boundary.
"""
struct FullCBE{T <: SweepDirection} <: CBEAlgorithm{T} 
     function FullCBE(direction::T = AnyDirection()) where T <: SweepDirection
          return new{T}()
     end
end

"""
     struct StandardCBE{T <: SweepDirection} <: CBEAlgorithm{T}
          D::Int64
          tol::Float64
     end

Standard CBE algorithm, details see [PhysRevLett.130.246402](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.246402).
"""
struct StandardCBE{T<:SweepDirection} <: CBEAlgorithm{T}
     D::Int64
     tol::Float64
     function StandardCBE(D::Int64, tol::Float64, direction::T = AnyDirection()) where T <: SweepDirection
          @assert D > 0
          @assert tol ≥ 0
          return new{T}(D, tol)
     end

end

# auto convert
function convert(::Type{CBEAlgorithm{T}}, Alg::NoCBE) where T <: Union{SweepL2R, SweepR2L}
     return NoCBE(T())
end
function convert(::Type{CBEAlgorithm{T}}, Alg::FullCBE) where T <: Union{SweepL2R, SweepR2L}
     return FullCBE(T())
end
function convert(::Type{CBEAlgorithm{T}}, Alg::StandardCBE{AnyDirection}) where T <: Union{SweepL2R, SweepR2L}
     return StandardCBE(Alg.D, Alg.tol, T())
end

"""
     struct CBEInfo{N}
          Alg::CBEAlgorithm
          info::NTuple{N, BondInfo}
          ϵ::Float64
     end

Information of CBE. `Alg` is the algorithm used. `info` contain the truncation info of `N` times svd. `ϵ = |Al*Ar - Al_ex*Ar_ex|` if calculated (otherwise `NaN`).
"""
struct CBEInfo{N}
     Alg::CBEAlgorithm
     info::NTuple{N, BondInfo}
     ϵ::Float64
     function CBEInfo(Alg::CBEAlgorithm, info::NTuple{N, BondInfo}, ϵ::Float64 = NaN) where N
          return new{N}(Alg, info, ϵ)
     end
end