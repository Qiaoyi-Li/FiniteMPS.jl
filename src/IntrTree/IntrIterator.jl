""" 
     abstract type AbstractInteractionIterator{L} 
The abstract type for the iterator of interaction terms. All the concrete subtypes should implement the `iterate` method so that the following usages are both valid where 
`Ops` is an instance of `AbstractInteractionIterator`.
```julia
for O in Ops
     # the i-th loop will give the i-th LocalOperator
end

collect(Ops) # return the vector of all LocalOperators 
```
"""
abstract type AbstractInteractionIterator{L} end

Base.IteratorSize(::Type{<:AbstractInteractionIterator}) = Base.HasLength()
Base.eltype(::Type{<:AbstractInteractionIterator}) = AbstractLocalOperator
length(::AbstractInteractionIterator{L}) where L = L

"""
     struct OnSiteInteractionIterator{L, T} <: AbstractInteractionIterator{L}
          Op::AbstractLocalOperator
          Z::T
     end
     
The iterator for on-site terms such as bosonic `n_i` or fermionic `c_i` together with its Jordan-Wigner string`Z_{i+1} Z_{i+2} ... Z_{L}`.

# Fields
     Op::AbstractLocalOperator
The local operator which tells both the operator and its site index.

     Z::Nothing 
For bosonic operators.
     Z::AbstractTensorMap
For fermionic operators. Assume all sites are fermionic therefore each site after `Op` will give `Z` operator.
     Z::AbstractVector{<:AbstractTensorMap}
Directly give the `Z` operator for each site to deal with the systems mixed with bosons and fermions.

# Constructors
     OnSiteInteractionIterator{L}(Op::AbstractLocalOperator, Z::T) where {L, T}
Direct constructor.

     OnSiteInteractionIterator{L}(Op::AbstractTensorMap,
          name::Union{String, Symbol},
          si::Int64;
          swap::Bool=false,
          Z=nothing)
Generate the `LocalOperator` object with `Op`, `name`, `si` and kwarg `swap`, details please see the constructors of `LocalOperator`. 
"""
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
          return OnSiteInteractionIterator{L}(LocalOperator(Op, name, si, !isnothing(Z); swap=swap), Z)
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
          Op_wrap = LocalOperator(iter.Z, :Z, i, false)
     else
          Op_wrap = IdentityOperator(i)
     end
     return Op_wrap, i + 1
end

"""
     struct TwoSiteInteractionIterator{L, T} <: AbstractInteractionIterator{L}
          O₁::AbstractLocalOperator
          O₂::AbstractLocalOperator
          Z::T
     end

The iterator for two-site terms such as bosonic `n_i n_j` or fermionic `c_i c_j` together with its Jordan-Wigner string `Z_{i+1} Z_{i+2} ... Z_{j}`.
     
Note the two site indices can be in both ascending and descending order but should not be the same. For the latter case, please use `OnSiteInteractionIterator` to get an on-site term instead.

# Fields
     O₁::AbstractLocalOperator
     O₂::AbstractLocalOperator
The local operators.

     Z::Nothing
For bosonic operators.
     Z::AbstractTensorMap
For fermionic operators. Note we assume both `O₁` and `O₂` are fermionic operators if `Z ≠ nothing`. 
     Z::AbstractVector{<:AbstractTensorMap}
Directly give the `Z` operator for each site to deal with the systems mixed with bosons and fermions.

# Constructors
     TwoSiteInteractionIterator{L}(O₁::AbstractLocalOperator, O₂::AbstractLocalOperator, Z::T) where {L, T}
Direct constructor.

     TwoSiteInteractionIterator{L}(Op::NTuple{2, AbstractTensorMap},
          name::NTuple{2, Union{String, Symbol}},
          si::NTuple{2, Int64};
          convertRight::Bool=false,
          Z=nothing)
Generate the `LocalOperator` objects with 2-tuples `Op`, `name` and `si`. If `convertRight == true`, convert the the second operator (with larger site index, can be `O₁` or `O₂`) to a right one (i.e. only have left horizontal leg), which is used to uniquely determine the contraction when calculating ITP. 
"""
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
          convertRight::Bool = false,
          Z=nothing
          ) where {L}
          @assert 1 ≤ si[1] ≤ L && 1 ≤ si[2] ≤ L && si[1] ≠ si[2]
          
          Zflag = !isnothing(Z)

          O₁ = LocalOperator(Op[1], name[1], si[1], Zflag)
          if si[1] < si[2]
               O₂ = LocalOperator(Op[2], name[2], si[2], Zflag)
          else
              # swap if si is not in ascending order
              fac = isnothing(Z) ? 1 : -1
              O₁, O₂ = _swap(O₁, LocalOperator(fac * Op[2], name[2], si[2], Zflag))
          end
          if convertRight
               # convert to right operator, i.e. the horizontal bond is on the left
               O₁, O₂ = _rightOp(O₁, O₂)
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
          Op_wrap = LocalOperator(iter.Z, :Z, i, false)
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
          @assert false "# TODO deal with the JW string"
          L = length(Op)
          Op = map(1:L) do i
               LocalOperator(Op[i], name, i)
          end
          return ArbitraryInteractionIterator{L}(Op)
     end
end
iterate(iter::ArbitraryInteractionIterator, args...) = iterate(iter.Op, args...) 

# make sure the additional horizontal bond is on the left, used in ITP
function _rightOp(A::LocalOperator{R₁, R₂}, B::LocalOperator{R₂, 1}) where {R₁, R₂}
     return A, B     
end
function _rightOp(A::LocalOperator{1, 2}, B::LocalOperator{2, 2})
     @tensor AB[f a b; d e] := A.A[a b c] * B.A[c d e f]  
     # QR
     TA, TB = leftorth(AB)
     return LocalOperator(permute(TA, ((1, 2), (3, 4))), A.name, A.si, A.fermionic, A.strength), LocalOperator(permute(TB, ((1, 2), (3, ))), B.name, B.si, B.fermionic, B.strength)
end

