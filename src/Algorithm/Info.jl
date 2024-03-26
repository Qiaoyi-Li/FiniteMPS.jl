"""
     struct LanczosInfo
          converged::Bool
          normres::Vector{Float64}
          numiter::Int64
          numops::Int64
     end

Similar to `KrylovKit.ConvergenceInfo` but delete `residuals` to save memory.
"""
struct LanczosInfo
     converged::Bool
     normres::Vector{Float64}
     numiter::Int64
     numops::Int64
end
function convert(::Type{LanczosInfo}, Info::KrylovKit.ConvergenceInfo)
     if isa(Info.normres, Float64)
          normres = [Info.normres,]
     else
          normres = Info.normres
     end
     return LanczosInfo(Info.converged > 0, normres, Info.numiter, Info.numops)
end

"""
     struct BondInfo
          D::Int64
          DD::Int64
          TrunErr::Float64
          SE::Float64
     end

Type for storing the information of a bond.

# Constructors
     BondInfo(s::AbstractTensorMap, ϵ::Float64 = 0.0)

Outer constructor via giving the `s` tensor and `ϵ` form `tsvd`.

     BondInfo(A::AbstractTensorMap, direction::Symbol)
     BondInfo(A::MPSTensor, direction::Symbol)

Outer constructor via giving a tensor `A` and `direction = :L` or `:R`. We cannot get truncation error and singular values hence `TrunErr` and `SE` are set to `0.0` and `NaN`, respectively.
"""
struct BondInfo
     D::Int64
     DD::Int64
     TrunErr::Float64
     SE::Float64
     BondInfo(D::Int64, DD::Int64, TrunErr::Float64, SE::Float64) = new(D, DD, TrunErr, SE)
end

function BondInfo(s::AbstractTensorMap{T}, ϵ::Float64=0.0) where T <: GradedSpace
     D = DD = 0
     Norm2 = SE = 0.0
     for k in keys(data(s))
          λ = diag(data(s)[k])
          D += length(λ)
          DD += length(λ) * dim(k)
          Norm2 += norm(λ)^2 * dim(k)
          SE += mapreduce(x -> x == 0 ? 0 : x^2 * log(x), +, λ) * dim(k)
     end
     SE = -2SE / Norm2 + log(Norm2)
     return BondInfo(D, DD, ϵ, SE)
end
function BondInfo(s::AbstractTensorMap{T}, ϵ::Float64=0.0) where T <: Union{CartesianSpace, ComplexSpace}
    D = DD = 0
    Norm2 = SE = 0.0

    λ = diag(data(s)[1])
    D += length(λ)
    DD += length(λ)
    Norm2 += norm(λ)^2
    SE += mapreduce(x -> x == 0 ? 0 : x^2 * log(x), +, λ; init = 0.0)

    SE = -2SE / Norm2 + log(Norm2)
    return BondInfo(D, DD, ϵ, SE)
end
function BondInfo(A::AbstractTensorMap, direction::Symbol)
     @assert direction in (:L, :R)
     idx = direction == :L ? 1 : rank(A)
     return BondInfo(dim(A, idx)..., 0.0, NaN)
end
BondInfo(A::MPSTensor, direction::Symbol) = BondInfo(A.A, direction)

function show(io::IO, info::BondInfo)
     print(io, "BondInfo(D = $(info.D) => $(info.DD), TrunErr2 = $(info.TrunErr^2), SE = $(info.SE))")
end

function merge(info1::BondInfo, info2::BondInfo)

     if isnan(info1.SE)
          SE = info2.SE
     elseif isnan(info2.SE)
          SE = info1.SE
     else
          SE = max(info1.SE, info2.SE)
     end

     return BondInfo(max(info1.D, info2.D),
          max(info1.DD, info2.DD),
          max(info1.TrunErr, info2.TrunErr),
          SE)
end
merge(info1::BondInfo, info2::BondInfo, args...) = merge(merge(info1, info2), args...)
merge(v::AbstractVector{BondInfo}) = reduce(merge, v)

"""
     struct DMRGInfo
          Eg::Float64
          Lanczos::LanczosInfo
          Bond::BondInfo
     end

Information of each DMRG update.
"""
struct DMRGInfo
     Eg::Float64
     Lanczos::LanczosInfo
     Bond::BondInfo
end

"""
     struct TDVPInfo{N,T}
          dt::T
          Lanczos::LanczosInfo
          Bond::BondInfo
     end

Information of each `N`-site TDVP update.

# Constructors
     TDVPInfo{N}(dt::Number, Lanczos::LanczosInfo, Bond::BondInfo)
"""
struct TDVPInfo{N,T}
     dt::T
     Lanczos::LanczosInfo
     Bond::BondInfo
     function TDVPInfo{N}(dt::Number, Lanczos::LanczosInfo, Bond::BondInfo) where {N}
          T = isa(dt, Real) ? Float64 : ComplexF64
          return new{N,T}(convert(T, dt), Lanczos, Bond)
     end
end
