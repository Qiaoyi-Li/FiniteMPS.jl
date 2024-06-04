abstract type AbstractInteractionIterator{L} end
Base.IteratorSize(::Type{<:AbstractInteractionIterator}) = Base.HasLength()
Base.eltype(::Type{<:AbstractInteractionIterator}) = AbstractLocalOperator
length(::AbstractInteractionIterator{L}) where L = L

struct OnSiteInteractionIterator{L, T} <: AbstractInteractionIterator{L}
     Op::AbstractLocalOperator
     Z::T
     function OnSiteInteractionIterator{L}(Op::AbstractLocalOperator,
          Z::T) where {L, T}
          @assert T <: Union{Nothing, AbstractTensorMap, AbstractVector{<:AbstractTensorMap}}
          if T <: AbstractVector{<:AbstractTensorMap}
               @assert length(Z) == L
          end
          return new{L, T}(Op, Z)
     end
     function OnSiteInteractionIterator{L}(Op::AbstractTensorMap,
          name::Union{String, Symbol},
          si::Int64;
          swap::Bool=false,
          Z = nothing) where {L}
          @assert 1 ≤ si ≤ L
          return OnSiteInteractionIterator{L}(LocalOperator(Op, name, si; swap=swap), Z)
     end
end
function iterate(iter::OnSiteInteractionIterator{L, Nothing}, i::Int64 = 1) where {L} 
     i > L && return nothing
     if i == iter.Op.si
          Op_wrap = iter.Op
     else
          Op_wrap = IdentityOperator(i)
     end
     return Op_wrap, i + 1
end
function iterate(iter::OnSiteInteractionIterator{L, <:AbstractTensorMap}, i::Int64 = 1) where {L} 
     i > L && return nothing
     if i == iter.Op.si
          Op_wrap = iter.Op
     elseif i > iter.Op.si
          Op_wrap = LocalOperator(iter.Z, :Z, i)
     else
          Op_wrap = IdentityOperator(i)
     end
     return Op_wrap, i + 1
end

struct TwoSiteInteractionIterator{L, T} <: AbstractInteractionIterator{L}
     O₁::AbstractLocalOperator
     O₂::AbstractLocalOperator
     Z::T
     function TwoSiteInteractionIterator{L}(O₁::AbstractLocalOperator,
          O₂::AbstractLocalOperator,
          Z::T) where {L, T}
          @assert T <: Union{Nothing, AbstractTensorMap, AbstractVector{<:AbstractTensorMap}}
          if T <: AbstractVector{<:AbstractTensorMap}
               @assert length(Z) == L
          end
          return new{L, T}(O₁, O₂, Z)
     end
     function TwoSiteInteractionIterator{L}(Op::NTuple{2, AbstractTensorMap},
          name::NTuple{2, Union{String, Symbol}},
          si::NTuple{2, Int64};
          Z=nothing) where {L}
          @assert 1 ≤ si[1] ≤ L && 1 ≤ si[2] ≤ L && si[1] ≠ si[2]
          
          O₁ = LocalOperator(Op[1], name[1], si[1])
          if si[1] < si[2]
               O₂ = LocalOperator(Op[2], name[2], si[2])
          else
              # swap if si is not in ascending order
              fac = isnothing(Z) ? 1 : -1
              O₁, O₂ = _swap(O₁, LocalOperator(fac * Op[2], name[2], si[2]))
          end
          return TwoSiteInteractionIterator{L}(O₁, O₂, Z)
     end
end
function iterate(iter::TwoSiteInteractionIterator{L, Nothing}, i::Int64 = 1) where {L} 
     i > L && return nothing
     if i == iter.O₁.si
          Op_wrap = iter.O₁
     elseif i == iter.O₂.si
          Op_wrap = iter.O₂
     else
          Op_wrap = IdentityOperator(i)
     end
     return Op_wrap, i + 1
end
function iterate(iter::TwoSiteInteractionIterator{L, <:AbstractTensorMap}, i::Int64 = 1) where {L} 
     i > L && return nothing
     if i == iter.O₁.si
          Op_wrap = iter.O₁
     elseif i == iter.O₂.si
          # add Z here 
          Op_wrap = _addZ!(iter.O₂, iter.Z) 
     elseif i > iter.O₁.si && i < iter.O₂.si
          Op_wrap = LocalOperator(iter.Z, :Z, i)
     else
          Op_wrap = IdentityOperator(i)
     end
     return Op_wrap, i + 1
end

struct ArbitraryInteractionIterator{L} <: AbstractInteractionIterator{L}
     Op::Vector{<:AbstractLocalOperator}
     function ArbitraryInteractionIterator{L}(Op::Vector{<:AbstractLocalOperator}) where L
          @assert length(Op) == L
          return new{L}(Op)
     end
     function ArbitraryInteractionIterator(Op::Vector{<:AbstractTensorMap},
          name::Union{String, Symbol})
          L = length(Op)
          Op = map(1:L) do i
               LocalOperator(Op[i], name, i)
          end
          return ArbitraryInteractionIterator{L}(Op)
     end
end
iterate(iter::ArbitraryInteractionIterator, args...) = iterate(iter.Op, args...) 
     

