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
          check::Bool
     end

Special case of CBE algorithm, directly keep the full bond space, usually used near boundary.

# Constructor
     FullCBE(direction::SweepDirection = AnyDirection(); check::Bool = false)
"""
struct FullCBE{T <: SweepDirection} <: CBEAlgorithm{T} 
     check::Bool
     function FullCBE(direction::T = AnyDirection(); check::Bool = false) where T <: SweepDirection
          return new{T}(check)
     end
end 

"""
     struct NaiveCBE{T<:SweepDirection} <: CBEAlgorithm{T}
          D::Int64
          tol::Float64
          rsvd::Bool
          check::Bool
     end

An naive implementation of CBE, where we directly contract the 2-site environment and then perform svd. Note the svd is O(D^3d^3) or O(D^3d^6) for MPS or MPO, respectively, which becomes the bottleneck of the algorithm. Therefore, a random svd is used to reduce the svd cost to O(D^3d) or O(D^3d^2).  

# Constructor
     NaiveCBE(D::Int64,
          tol::Float64,
          direction::SweepDirection = AnyDirection();
          rsvd::Bool = false,
          check::Bool = false)
"""
struct NaiveCBE{T<:SweepDirection} <: CBEAlgorithm{T}
     D::Int64
     tol::Float64
     rsvd::Bool
     check::Bool
     function NaiveCBE(D::Int64, tol::Float64, direction::T = AnyDirection(); rsvd::Bool = false, check::Bool = false) where T <: SweepDirection
          @assert D > 0
          @assert tol ≥ 0
          return new{T}(D, tol, rsvd, check)
     end
end


# auto convert
function convert(::Type{CBEAlgorithm{T}}, Alg::NoCBE) where T <: Union{SweepL2R, SweepR2L}
     return NoCBE(T())
end
function convert(::Type{CBEAlgorithm{T}}, Alg::FullCBE) where T <: Union{SweepL2R, SweepR2L}
     return FullCBE(T(); check=Alg.check)
end
function convert(::Type{CBEAlgorithm{T}}, Alg::NaiveCBE{AnyDirection}) where T <: Union{SweepL2R, SweepR2L}
     return NaiveCBE(Alg.D, Alg.tol, T(); check=Alg.check, rsvd = Alg.rsvd)
end


"""
     struct CBEInfo{N}
          Alg::CBEAlgorithm
          info::NTuple{N, BondInfo}
          D₀::NTuple{2, Int64}
          D::NTuple{2, Int64}
          ϵp::Float64
          ϵ::Float64
     end

Information of CBE. `Alg` is the algorithm used. `info` contain the truncation info of `N` times svd. `D₀` and `D` are the initial and final bond dimension, respectively. `ϵp` is the estimated projection error. `ϵ = |Al*Ar - Al_ex*Ar_ex|` if calculated (otherwise `NaN`). 
"""
struct CBEInfo{N}
     Alg::CBEAlgorithm
     info::NTuple{N, BondInfo}
     D₀::NTuple{2, Int64}
     D::NTuple{2, Int64}
     ϵp::Float64
     ϵ::Float64
     function CBEInfo(Alg::CBEAlgorithm,
          info::NTuple{N, BondInfo},
          D₀::NTuple{2, Int64},
          D::NTuple{2, Int64},
          ϵp::Float64,
          ϵ::Float64 = NaN) where N
          return new{N}(Alg, info, D₀, D, ϵp, ϵ)
     end
end