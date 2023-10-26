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
     return LanczosInfo(Info.converged > 0, Info.normres, Info.numiter, Info.numops)
end

"""
     struct BondInfo
          D::Int64
          DD::Int64
          TrunErr::Float64
          SE::Float64
     end

Type for storing the information of a bond.
"""
struct BondInfo
     D::Int64
     DD::Int64
     TrunErr::Float64
     SE::Float64
     BondInfo(D::Int64, DD::Int64, TrunErr::Float64, SE::Float64) = new(D, DD, TrunErr, SE)
end

"""
     BondInfo(s::AbstractTensorMap, ϵ::Float64 = 0.0)

Outer constructor via giving the `s` tensor and `ϵ` form `tsvd`.

     BondInfo(A::AbstractTensorMap, direction::Symbol)
     BondInfo(A::MPSTensor, direction::Symbol)

Outer constructor via giving a tensor `A` and `direction = :L` or `:R`. We cannot get truncation error and singular values hence `TrunErr` and `SE` are set to `0.0` and `NaN`, respectively.  
"""
function BondInfo(s::AbstractTensorMap, ϵ::Float64=0.0)
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
function BondInfo(A::AbstractTensorMap, direction::Symbol)
     @assert direction in (:L, :R)
     idx = direction == :L ? 1 : rank(A)
     return BondInfo(dim(A, idx)..., 0.0, NaN)
end
BondInfo(A::MPSTensor, direction::Symbol) = BondInfo(A.A, direction)


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