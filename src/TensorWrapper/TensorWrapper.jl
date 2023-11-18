"""
     abstract type AbstractTensorWrapper

Wrapper type for classifying different Tensors.

Note each concrete subtype must have a field `A::AbstractTensorMap` to save the Tensor.
"""
abstract type AbstractTensorWrapper end

# some common functions for wrapper type
convert(::Type{T}, A::AbstractTensorMap) where {T<:AbstractTensorWrapper} = T(A)
for func in (:dim, :rank, :domain, :codomain, :eltype, :norm, :scalartype, :data)
     # Tensor -> Number
     @eval $func(obj::AbstractTensorWrapper, args...) = $func(obj.A, args...)
end
for func in (:similar, :one, :zero)
     # Tensor -> Tensor(wrapped)
     @eval $func(obj::T) where {T<:AbstractTensorWrapper} = convert(T, $func(obj.A))
end
for func in (:leftnull, :rightnull)
     # Tensor -> Tensor
     @eval $func(obj::T, args...; kwargs...) where {T<:AbstractTensorWrapper} = $func(obj.A, args...; kwargs...)
end

for func in (:dot, :inner)
     # Tensor × Tensor -> Number
     @eval $func(A::T, B::T) where {T<:AbstractTensorWrapper} = $func(A.A, B.A)
end
function normalize!(A::AbstractTensorWrapper)
     normalize!(A.A)
     return A
end

==(A::AbstractTensorWrapper, B::AbstractTensorWrapper) = A.A == B.A

*(A::AbstractTensorWrapper, B::AbstractTensorWrapper) = A.A * B.A

# linear algebra 
+(A::T, B::T) where {T<:AbstractTensorWrapper} = convert(T, A.A + B.A)
+(A::AbstractTensorWrapper, ::Nothing) = A
+(::Nothing, A::AbstractTensorWrapper) = A
-(A::T) where {T<:AbstractTensorWrapper} = convert(T, -A.A)
-(A::T, B::T) where {T<:AbstractTensorWrapper} = convert(T, A.A - B.A)
-(A::AbstractTensorWrapper, ::Nothing) = A
-(::Nothing, A::AbstractTensorWrapper) = -A

*(A::T, a::Number) where {T<:AbstractTensorWrapper} = convert(T, a * A.A)
*(a::Number, A::AbstractTensorWrapper) = A * a

/(A::T, a::Number) where {T<:AbstractTensorWrapper} = convert(T, A.A / a)

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
axpy!(::Number, ::Nothing, A::AbstractTensorWrapper) = A
axpy!(α::Number, A::AbstractTensorWrapper, ::Nothing) = α * A
function axpby!(α::Number, A::T, β::Number, B::T) where {T<:AbstractTensorWrapper}
     axpby!(α, A.A, β, B.A)
     return B
end
axpby!(α::Number, ::Nothing, β::Number, A::AbstractTensorWrapper) = rmul!(A, β)
axpby!(α::Number, A::AbstractTensorWrapper, β::Number, ::Nothing) = axpy!(α, A, nothing)

"""
     tsvd(A::AbstractTensorWrapper,
          p₁::NTuple{N₁,Int64},
          p₂::NTuple{N₂,Int64};
          kwargs...) 
          -> u::AbstractTensorMap, s::AbstractTensorMap, vd::AbstractTensorMap, info::BondInfo  

Wrap TensorKit.tsvd, return `BondInfo` struct instead of truncation error `ϵ`.
"""
function tsvd(A::AbstractTensorWrapper, p₁::NTuple{N₁,Int64}, p₂::NTuple{N₂,Int64}; kwargs...) where {N₁,N₂}
     trunc = get(kwargs, :trunc, notrunc())
     alg = get(kwargs, :alg, nothing)
     if isnothing(alg)
          u,s,v,ϵ = tsvd(A.A, p₁, p₂; trunc=trunc)
     else
          u,s,v,ϵ = tsvd(A.A, p₁, p₂; trunc=trunc, alg=alg)
     end
     return u, s, v, BondInfo(s, ϵ)
end
tsvd(A::AbstractTensorWrapper, p::Index2Tuple; kwargs...) = tsvd(A, p[1], p[2]; kwargs...)








