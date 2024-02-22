"""
     struct CompositeMPSTensor{N, T <: NTuple{N, MPSTensor}} <: AbstractMPSTensor
          A::AbstractTensorMap
     end

Wrapper type for multi-site local tensors.

The 2nd parameter indicates the types of the N original on-site tensors.

# Constructors
     CompositeMPSTensor{N, T}(::AbstractTensorMap) where T <: NTuple{N, MPSTensor}
Directly construct.

     CompositeMPSTensor(::NTuple{N, MPSTensor})
     CompositeMPSTensor(::MPSTensor, ::MPSTensor, ...)
Contract N on-site tensors to get the N-site tensor. 
"""
struct CompositeMPSTensor{N,T<:NTuple{N,MPSTensor}} <: AbstractMPSTensor
     A::AbstractTensorMap

     function CompositeMPSTensor{N,T}(A::AbstractTensorMap) where {N,T<:NTuple{N,MPSTensor}}
          return new{N,T}(A)
     end

     function CompositeMPSTensor(x::T) where {N,T<:NTuple{N,MPSTensor}}
          @assert N ≥ 2
          A = reduce(*, x)
          return new{N,T}(A)
     end
     CompositeMPSTensor(x::MPSTensor...) = CompositeMPSTensor(x)
end

"""
     leftorth(::CompositeMPSTensor{2, ...};
          trunc = notrunc(),
          kwargs...) -> Q::AbstractTensorMap, R::AbstractTensorMap, info::BondInfo

Split a 2-site local tensor s.t. the left one is canonical.
"""
function leftorth(A::CompositeMPSTensor{2,Tuple{MPSTensor{R₁},MPSTensor{R₂}}}; trunc=notrunc(), kwargs...) where {R₁,R₂}
     if trunc == notrunc()
          Q, R = leftorth(A.A, Tuple(1:R₁-1), Tuple(R₁ - 1 .+ (1:R₂-1)))
          return Q, R, BondInfo(Q, :R)
     else
          u, s, vd, info = tsvd(A; trunc=trunc, kwargs...)
          return u, s * vd, info
     end
end

"""
     rightorth(::CompositeMPSTensor{2, ...};
          trunc = notrunc(),
          kwargs...) -> L::AbstractTensorMap, Q::AbstractTensorMap, info::BondInfo

Split a 2-site local tensor s.t. the right one is canonical.
"""
function rightorth(A::CompositeMPSTensor{2,Tuple{MPSTensor{R₁},MPSTensor{R₂}}}; trunc=notrunc(), kwargs...) where {R₁,R₂}
     if trunc == notrunc()
          L, Q = rightorth(A.A, Tuple(1:R₁-1), Tuple(R₁ - 1 .+ (1:R₂-1)))
          return L, Q, BondInfo(Q, :L)
     else
          u, s, vd, info = tsvd(A; trunc=trunc, kwargs...)
          return u * s, vd, info
     end
end

"""
     tsvd(::CompositeMPSTensor{2, ...}; kwargs...) 
          -> u::AbstractTensorMap, s::AbstractTensorMap, vd::AbstractTensorMap, info::BondInfo

Use SVD to split a 2-site local tensor, details see TensorKit.tsvd.
"""
function tsvd(A::CompositeMPSTensor{2,Tuple{MPSTensor{R₁},MPSTensor{R₂}}}; kwargs...) where {R₁,R₂}
     return tsvd(A, Tuple(1:R₁-1), Tuple(R₁ - 1 .+ (1:R₂-1)); kwargs...)
end


"""
     noise!(A::CompositeMPSTensor{2}, σ::Real)

Apply noise to a given 2-site local tensor by contracting a `d×d` random isometry to it. 
"""
function noise!(A::CompositeMPSTensor{2,Tuple{MPSTensor{R₁},MPSTensor{R₂}}}, σ::Real) where {R₁,R₂}
     perms = vcat(2, R₁, setdiff(1:R₁+R₂-2, [2, R₁]))
     iperms = invperm(perms)
     RA = permute(A.A, (Tuple(perms[1:2]), Tuple(perms[3:end])))

     Iso = randisometry(eltype(RA), codomain(RA); σ=σ)
     permute!(A.A, Iso * RA, (Tuple(iperms[1:R₁-1]), Tuple(iperms[R₁:end])))
     return A
end

