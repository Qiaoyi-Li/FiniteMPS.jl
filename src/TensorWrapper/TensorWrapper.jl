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


*(A::AbstractTensorWrapper, B::AbstractTensorWrapper) = A.A * B.A

# linear algebra 
+(A::T, B::T) where {T<:AbstractTensorWrapper} = convert(T, A.A + B.A)
+(A::AbstractTensorWrapper, ::Nothing) = A
+(::Nothing, A::AbstractTensorWrapper) = A
+(::Nothing, ::Nothing) = nothing
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
add!(A::AbstractTensorWrapper, B::AbstractTensorWrapper) = axpy!(true, B, A)
add!(A::AbstractTensorWrapper, ::Nothing) = A
add!(::Nothing, A::AbstractTensorWrapper) = A

# add methods for vectorinterface.jl, which is used in KrylovKit after v0.7
function add!!(A::AbstractTensorWrapper,
     B::AbstractTensorWrapper,
     β::Number = one(scalartype(B)),
     α::Number = one(scalartype(A))
     ) 
     T = promote_type(scalartype(A.A), scalartype(B.A), typeof(α), typeof(β))
     if T <: scalartype(A.A)
          return axpby!(β, B, α, A)
     else
          return α*A + β*B
     end
end
function zerovector(A::T, ::Type{S}) where {S<:Number, T<:AbstractTensorWrapper}
     return convert(T, zerovector(A.A, S))
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
     if T <: scalartype(A)
          return scale!(A, α)
     else
          return scale(A, α)
     end
end

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
     alg = get(kwargs, :alg, SDD())
     p = get(kwargs, :p, 2)

     u,s,v,ϵ = _tsvd_try(A.A, (p₁, p₂); p = p, trunc=trunc, alg=alg)

     return u, s, v, BondInfo(s, ϵ)
end
tsvd(A::AbstractTensorWrapper, p::Index2Tuple; kwargs...) = tsvd(A, p[1], p[2]; kwargs...)

# use try to avoid errors in tsvd by SDD()
function _tsvd_try(t::AbstractTensorMap;
     trunc::TruncationScheme=NoTruncation(),
     p::Real=2, alg::Union{SVD,SDD}=SDD())

     try
          return tsvd!(copy(t); trunc=trunc, p=p, alg=alg)
     catch
          @assert alg = SDD()
          @warn "SDD() failed, use SVD() instead."
          return tsvd!(copy(t); trunc=trunc, p=p, alg=SVD())
     end
end
function _tsvd_try(t::AbstractTensorMap,
     (p₁, p₂)::Index2Tuple; kwargs...)
     try
         return tsvd!(permute(t, (p₁, p₂); copy=true); kwargs...)
     catch
         @assert get(kwargs, :alg, SDD()) == SDD()
         @warn "SDD() failed, use SVD() instead."
         trunc = get(kwargs, :trunc, TensorKit.NoTruncation())
         p = get(kwargs, :p, 2)
         return tsvd!(permute(t, (p₁, p₂); copy=true); trunc=trunc, p=p, alg=SVD())
     end
 end







