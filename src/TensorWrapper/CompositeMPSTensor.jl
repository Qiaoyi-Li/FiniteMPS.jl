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
          kwargs...) -> Q::AbstractTensorMap, R::AbstractTensorMap, ϵ::Float64

Split a 2-site local tensor s.t. the left one is canonical.
"""
function leftorth(A::CompositeMPSTensor{2,Tuple{MPSTensor{R₁},MPSTensor{R₂}}}; trunc=notrunc(), kwargs...) where {R₁,R₂}
     if trunc == notrunc()
          return leftorth(A.A, Tuple(1:R₁-1), Tuple(R₁ - 1 .+ (1:R₂-1)))..., 0.0
     else
          u, s, vd, ϵ = tsvd(A; trunc=trunc, kwargs...)
          return u, s * vd, ϵ
     end
end

"""
     rightorth(::CompositeMPSTensor{2, ...};
          trunc = notrunc(),
          kwargs...) -> L::AbstractTensorMap, Q::AbstractTensorMap, ϵ::Float64

Split a 2-site local tensor s.t. the right one is canonical.
"""
function rightorth(A::CompositeMPSTensor{2,Tuple{MPSTensor{R₁},MPSTensor{R₂}}}; trunc=notrunc(), kwargs...) where {R₁,R₂}
     if trunc == notrunc()
          return rightorth(A.A, Tuple(1:R₁-1), Tuple(R₁ - 1 .+ (1:R₂-1)))..., 0.0
     else
          u, s, vd, ϵ = tsvd(A; trunc=trunc, kwargs...)
          return u * s, vd, ϵ
     end
end

"""
     tsvd(::CompositeMPSTensor{2, ...}; kwargs...) 
          -> u::AbstractTensorMap, s::AbstractTensorMap, vd::AbstractTensorMap, ϵ::Float64

Use SVD to split a 2-site local tensor, details see TensorKit.tsvd.
"""
function tsvd(A::CompositeMPSTensor{2,Tuple{MPSTensor{R₁},MPSTensor{R₂}}}; kwargs...) where {R₁,R₂}
     trunc = get(kwargs, :trunc, notrunc())
     alg = get(kwargs, :alg, nothing)
     if isnothing(alg)
          return tsvd(A.A, Tuple(1:R₁-1), Tuple(R₁ - 1 .+ (1:R₂-1)); trunc=trunc)
     else
          return tsvd(A.A, Tuple(1:R₁-1), Tuple(R₁ - 1 .+ (1:R₂-1)); trunc=trunc, alg=alg)
     end
end