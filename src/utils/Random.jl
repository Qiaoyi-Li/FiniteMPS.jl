"""
     abstract type AbstractDistribution{F<:Union{Float64, ComplexF64}}

Abstract type of probability distributions on Stiefel manifold `Vₙₖ`.
"""
abstract type AbstractDistribution{F<:Union{Float64,ComplexF64}} end

"""
     struct UniformDistribution{F} <: AbstractDistribution{F}

Type of the uniform distribution on Stiefel manifold `Vₙₖ`, induced by the Haar measure on `O(n) == Vₙₙ`.
"""
struct UniformDistribution{F} <: AbstractDistribution{F}
     function UniformDistribution{F}() where {F}
          return new{F}()
     end
     UniformDistribution() = UniformDistribution{Float64}()
end

"""
     struct GaussianDistribution{F} <: AbstractDistribution{F}
          σ::Float64
     end

Type of the gaussian distribution around `[Iₖₖ, 0]` on Stiefel manifold `Vₙₖ`. 

Note this is not strictly gaussian distribution on `Vₙₖ`, in practical we will first generate a gaussian distribution `N(0,σ)` on `so(n)` and then `exp` to `SO(n)`.
"""
struct GaussianDistribution{F} <: AbstractDistribution{F}
     σ::Float64
     function GaussianDistribution{F}(σ::Real) where {F}
          @assert σ > 0
          return new{F}(σ)
     end
end

"""
     const NormalDistribution = GaussianDistribution 
"""
const NormalDistribution = GaussianDistribution

"""
     randStiefel([Dist::UniformDistribution, ] n::Int64, k::Int64) -> ::Matrix (n × k)
     randStiefel(Dist::GaussianDistribution) -> ::Matrix (n × k)

Random sample on Stiefel manifold `Vₙₖ` with given distribution.

Default `Dist = UniformDistribution{Float64}`.
"""
function randStiefel(Dist::UniformDistribution{F}, n::Int64, k::Int64=n) where {F}
     n < k && return hcat(randStiefel(Dist, n), zeros(F, n, k-n))

     q::Matrix, _ = qr(randn(F, n, k))
     # we find qr will choose a fixed orientation in some cases
     q[:, 1] = view(q, :, 1) * _rand_base(F)
     return q
end
randStiefel(::Type{F}, n::Int64, k::Int64=n) where {F<:Union{Float64,ComplexF64}} = randStiefel(UniformDistribution{F}(), n, k)
randStiefel(n::Int64, k::Int64=n) = randStiefel(Float64, n, k)

# base case, O(1) or U(1)
_rand_base(::Type{Float64}) = rand([1, -1])
_rand_base(::Type{ComplexF64}) = exp(2pi * im * rand(Float64))

function randStiefel(Dist::GaussianDistribution{F}, n::Int64, k::Int64=n) where {F}
     n < k && return hcat(randStiefel(Dist, n), zeros(F, n, k-n))

     g = _randson(F, n, Dist.σ)
     return exp(g)[:, 1:k]
end
function _randson(::Type{F}, n::Int64, σ::Float64) where {F<:Union{Float64,ComplexF64}}
     g = zeros(F, n, n)
     for i = 1:n, j = i+1:n
          g[i, j] = randn(F) * σ
          g[j, i] = -conj(g[i, j])
     end
     return g
end


"""  
     randisometry([::Type{T},] codom::VectorSpace, dom::VectorSpace = codom; kwargs...)
     randisometry([::Type{T},] A::AbstractTensorMap; kwargs...)

Generate random tensors based on `randStiefel`. Valid kwargs please see the mutating version `randisometry!`.
"""
function randisometry(::Type{T}, codom::VectorSpace, dom::VectorSpace = codom; kwargs...) where {T<:Union{Float64,ComplexF64}}
     A = TensorMap(zeros, T, codom, dom)
     return randisometry!(A; kwargs...)
end
function randisometry(::Type{T}, A::AbstractTensorMap; kwargs...) where {T<:Union{Float64,ComplexF64}}
     return randisometry!(similar(A, T); kwargs...)
end
randisometry(A::AbstractTensorMap; kwargs...) = randisometry(eltype(A), A; kwargs...)

"""
     randisometry!(A::AbstractTensorMap; kwargs...) 

Randomize the data of tensor `A` based on `randStiefel`.

# Keywords
     Dist => ::Symbol
`:Uniform` or `:Gaussian` ( == `:Normal`). If `σ` is given, default = `:Gaussian`, otherwise `:Uniform`. 

     σ => ::Real
`σ` of gaussian distribution. No default value. It will throw an error if `σ` is not given and `Dist = :Gaussian`.
"""
function randisometry!(A::AbstractTensorMap; kwargs...)
     Data = data(A)
     for k in keys(Data)
          _rand_data!(Data[k]; kwargs...)
     end
     return A
end

function _rand_data!(A::AbstractMatrix{T}; kwargs...) where {T}

     SymDist = get(kwargs, :Dist, :σ ∈ keys(kwargs) ? :Gaussian : :Uniform)
     if SymDist == :Uniform
          Dist = UniformDistribution{T}()
     else
          @assert SymDist ∈ [:Gaussian, :Normal]
          @assert :σ ∈ keys(kwargs)
          σ = get(kwargs, :σ, nothing)
          Dist = GaussianDistribution{T}(σ)
     end
     
     A[:] = randStiefel(Dist, size(A)...)
     return A
end


