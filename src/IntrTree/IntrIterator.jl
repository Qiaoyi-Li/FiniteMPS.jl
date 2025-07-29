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
          pspace = domain(iter.Op)[1]
          Op_wrap = IdentityOperator(pspace, trivial(pspace), i)
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
          pspace = domain(iter.Op)[1]
          Op_wrap = IdentityOperator(pspace, trivial(pspace), i)
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
function iterate(iter::TwoSiteInteractionIterator{L, Nothing}, st::Tuple{Int64, VectorSpace} = (1, getLeftSpace(iter.O₁))) where {L} 
     i, aspace = st
     i > L && return nothing
     if i == iter.O₁.si
          Op_wrap = iter.O₁
     elseif i == iter.O₂.si
          Op_wrap = iter.O₂
     else
          # deduce pspace 
          pspace = domain(iter.O₁)[1]
          Op_wrap = IdentityOperator(pspace, aspace, i)
     end
     return Op_wrap, (i + 1, getRightSpace(Op_wrap))
end
function iterate(iter::TwoSiteInteractionIterator{L, <:AbstractTensorMap}, st::Tuple{Int64, VectorSpace} = (1, getLeftSpace(iter.O₁))) where {L} 
     i, aspace = st
     i > L && return nothing
     if i == iter.O₁.si
          Op_wrap = iter.O₁
     elseif i == iter.O₂.si
          # add Z here 
          Op_wrap = _addZ!(iter.O₂, iter.Z) 
     elseif i > iter.O₁.si && i < iter.O₂.si
          Op_wrap = LocalOperator(iter.Z, :Z, i, false; aspace = (aspace, aspace))
     else
          pspace = domain(iter.Z)[1]
          Op_wrap = IdentityOperator(pspace, aspace, i)
     end

     return Op_wrap, (i + 1, getRightSpace(Op_wrap))
end

"""
     ArbitraryInteractionIterator{L} <: AbstractInteractionIterator{L}
          Ops::Vector{<:AbstractLocalOperator}
          Z::Union{Nothing, AbstractTensorMap, AbstractVector{<:AbstractTensorMap}}
          pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}}
     end

The iterator for an arbitrary interaction term. 

# Fields
     Ops::Vector{<:AbstractLocalOperator}
A vector to store the local operators, `length(Ops) == N` means a `N`-site interaction term.

     Z::Union{Nothing, AbstractTensorMap, AbstractVector{<:AbstractTensorMap}}
     pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}}
Provide the fermion parity operator `Z` and the local physical space `pspace`. Assume a site-independent `Z` or `pspace` if a single object is provided, otherwise, a length-`L` vector is expected to deal with the site-dependent cases.  

# Constructors
     ArbitraryInteractionIterator{L}(Ops::Vector{<:AbstractLocalOperator},
          Z::Union{Nothing, AbstractTensorMap, AbstractVector{<:AbstractTensorMap}},
          pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing)
The direct constructor. 
"""
struct ArbitraryInteractionIterator{L} <: AbstractInteractionIterator{L}
     Ops::Vector{<:AbstractLocalOperator}
     Z::Union{Nothing, AbstractTensorMap, AbstractVector{<:AbstractTensorMap}}
     pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}}
end
function iterate(iter::ArbitraryInteractionIterator{L}, st::Tuple{Int64, Int64, Bool, VectorSpace} = (1, 1, false, getLeftSpace(iter.Ops[1]))) where {L}
     i, idx, flag, aspace = st
     i > L && return nothing

     if idx > length(iter.Ops) || i < iter.Ops[idx].si
          if flag 
               return LocalOperator(_getZ(iter.Z, i), :Z, i, false; aspace = (aspace, aspace)), (i + 1, idx, flag, aspace)
          else
               return IdentityOperator(_getpspace(iter.pspace, i), aspace, i), (i + 1, idx, flag, aspace)
          end
     else
          Op_i = deepcopy(iter.Ops[idx])
          # add Z if necessary 
          flag && _addZ!(Op_i, _getZ(iter.Z, i))
          return Op_i, (i + 1, idx + 1, xor(flag, isfermionic(Op_i)), getRightSpace(Op_i))
     end
end

_getZ(::Nothing, ::Int64) = nothing
_getZ(Z::AbstractTensorMap, ::Int64) = Z
_getZ(Z::Vector{AbstractTensorMap}, i::Int64) = Z[i]
_getpspace(::Nothing, ::Int64) = nothing
_getpspace(pspace::VectorSpace, ::Int64) = pspace
_getpspace(pspace::Vector{VectorSpace}, i::Int64) = pspace[i]

_addZ!(O::LocalOperator, ::Nothing) = O
function _addZ!(OR::LocalOperator{1, 1}, Z::AbstractTensorMap)
	OR.A = Z * OR.A
	if OR.name[1] == 'Z'
		OR.name = OR.name[2:end]
	else
		OR.name = "Z" * OR.name
	end
	return OR
end
function _addZ!(OR::LocalOperator{1, R₂}, Z::AbstractTensorMap) where R₂
	OR.A = Z * OR.A
	if OR.name[1] == 'Z'
		OR.name = OR.name[2:end]
	else
		OR.name = "Z" * OR.name
	end
	return OR
end
function _addZ!(OR::LocalOperator{R₁, 1}, Z::AbstractTensorMap) where R₁
	# note ZA = - AZ
	OR.A = -OR.A * Z
	if OR.name[1] == 'Z'
		OR.name = OR.name[2:end]
	else
		OR.name = "Z" * OR.name
	end
	return OR
end

function _addZ!(OR::LocalOperator{2, 2}, Z::AbstractTensorMap)
	@tensor OR.A[a e; c d] := Z[e b] * OR.A[a b c d]
	if OR.name[1] == 'Z'
		OR.name = OR.name[2:end]
	else
		OR.name = "Z" * OR.name
	end
	return OR
end

