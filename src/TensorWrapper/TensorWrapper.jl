"""
     abstract type AbstractTensorWrapper

Wrapper type for classifying different Tensors.

Note each concrete subtype must have a field `A::AbstractTensorMap` to save the Tensor.
"""
abstract type AbstractTensorWrapper end

# some common functions for wrapper type
convert(::Type{T}, A::AbstractTensorMap) where {T<:AbstractTensorWrapper} = T(A)
_rewrap(obj::T, A::AbstractTensorMap) where {T<:AbstractTensorWrapper} = T(A)
for func in (:dim, :bonddim, :rank, :domain, :codomain, :eltype, :norm, :scalartype, :numin, :numout, :numind)
     # Tensor -> Number
     @eval $func(obj::AbstractTensorWrapper, args...) = $func(obj.A, args...)
end
for func in (:similar, :one, :zero)
     # Tensor -> Tensor(wrapped)
     @eval $func(obj::AbstractTensorWrapper) = _rewrap(obj, $func(obj.A))
end

for func in (:dot, :inner)
     # Tensor × Tensor -> Number
     @eval $func(A::T, B::T) where {T<:AbstractTensorWrapper} = $func(A.A, B.A)
end
function normalize!(A::AbstractTensorWrapper)
     normalize!(A.A)
     return A
end

*(A::AbstractTensorWrapper, B::AbstractTensorWrapper) = A.A * B.A

# linear algebra
+(A::T, B::T) where {T<:AbstractTensorWrapper} = _rewrap(A, A.A + B.A)
-(A::AbstractTensorWrapper) = _rewrap(A, -A.A)
-(A::T, B::T) where {T<:AbstractTensorWrapper} = _rewrap(A, A.A - B.A)
*(A::AbstractTensorWrapper, a::Number) = _rewrap(A, a * A.A)
*(a::Number, A::AbstractTensorWrapper) = A * a
/(A::AbstractTensorWrapper, a::Number) = _rewrap(A, A.A / a)

function mul!(A::T, B::T, α::Number) where {T<:AbstractTensorWrapper}
     mul!(A.A, B.A, α)
     return A
end
function rmul!(A::AbstractTensorWrapper, α::Number)
     rmul!(A.A, α)
     return A
end
function axpy!(α::Number, A::T, B::T) where {T<:AbstractTensorWrapper}
     axpy!(α, A.A, B.A)
     return B
end
function axpby!(α::Number, A::T, β::Number, B::T) where {T<:AbstractTensorWrapper}
     axpby!(α, A.A, β, B.A)
     return B
end
function add!(y::W, x::W, α::Number, β::Number) where {W<:AbstractTensorWrapper}
     add!(y.A, x.A, α, β)
     return y
end

# add methods for vectorinterface.jl, which is used in KrylovKit after v0.7
function add!!(y::W, x::W, α::Number, β::Number) where {W<:AbstractTensorWrapper}
     A = add!!(y.A, x.A, α, β)
     return A === y.A ? y : _rewrap(y, A)
end
function zerovector(A::AbstractTensorWrapper, ::Type{S}) where {S<:Number}
     return _rewrap(A, zerovector(A.A, S))
end
function zerovector!(A::AbstractTensorWrapper)
     zerovector!(A.A)
     return A
end
similar(A::AbstractTensorWrapper, ::Type{S}) where {S<:Number} = zerovector(A, S)
scale!(A::AbstractTensorWrapper, α::Number) = rmul!(A, α)
scale(A::AbstractTensorWrapper, α::Number) = α * A
function scale!!(A::AbstractTensorWrapper, α::S) where {S<:Number}
     T = promote_type(scalartype(A.A), S)
     return T <: scalartype(A) ? scale!(A, α) : scale(A, α)
end

"""
     tsvd(A::AbstractTensorWrapper,
          p₁::NTuple{N₁,Int64},
          p₂::NTuple{N₂,Int64};
          kwargs...)
          -> u::AbstractTensorMap, s::DiagonalTensorMap, vd::AbstractTensorMap, info::BondInfo

Compute a compact or truncated SVD, returning `BondInfo` instead of the 2-norm truncation error. Only `p=2` is supported.
"""
function tsvd(A::AbstractTensorWrapper, p₁::NTuple{N₁,Int64}, p₂::NTuple{N₂,Int64}; kwargs...) where {N₁,N₂}
     get(kwargs, :p, 2) == 2 || throw(ArgumentError("tsvd supports only p=2"))
     trunc = get(kwargs, :trunc, notrunc())
     alg = get(kwargs, :alg, MatrixAlgebraKit.DivideAndConquer(fixgauge = false))
     u, s, v, ϵ = _raw_svd(A.A, (p₁, p₂); trunc, alg)
     return u, s, v, BondInfo(s, ϵ)
end
tsvd(A::AbstractTensorWrapper, p::Index2Tuple; kwargs...) = tsvd(A, p[1], p[2]; kwargs...)
