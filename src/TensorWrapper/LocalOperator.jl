"""
	 abstract type AbstractLocalOperator <: AbstractTensorWrapper

Wrapper type for classifying differnet local operators in order to accelerate contractions.
"""
abstract type AbstractLocalOperator <: AbstractTensorWrapper end

"""
	 hastag(::AbstractLocalOperator) -> ::Bool

Check if this `LocalOperator` has field `tag` or not.
"""
hastag(::AbstractLocalOperator) = false

function rmul!(O::AbstractLocalOperator, α::Number)
	# change strength instead of the tensor
	O.strength *= α
	return O
end

"""
	 mutable struct IdentityOperator <: AbstractLocalOperator
		  pspace::Union{Nothing, VectorSpace}
		  si::Int64
		  strength::Number 
	 end
	 
Lazy type of identity operator, used for skipping some tensor contractions.

# Constructors
	 IdentityOperator([pspace::VectorSpace,] si::Int64, strength::Number = NaN)
"""
mutable struct IdentityOperator <: AbstractLocalOperator
	pspace::Union{Nothing, VectorSpace}
	si::Int64
	strength::Number
	function IdentityOperator(si::Int64, strength::Number = NaN)
		return new(nothing, si, strength)
	end
	function IdentityOperator(::Nothing, si::Int64, strength::Number = NaN)
		return new(nothing, si, strength)
	end
	function IdentityOperator(pspace::VectorSpace, si::Int64, strength::Number = NaN)
		return new(pspace, si, strength)
	end
end

"""
	 getOpName(::AbstractLocalOperator) -> ::String

Interface of `AbstractLocalOperator`, return `"I"` for `IdentityOperator` and `O.name` for `O::LocalOperator`.
"""
getOpName(::IdentityOperator) = "I"

"""
	isfermionic(::AbstractLocalOperator) -> ::Bool

Return the boolean value indicating whether the operator is fermionic or not.
"""
isfermionic(::IdentityOperator) = false

function Base.show(io::IO, obj::IdentityOperator)
	print(io, "I$(String(collect("$(obj.si)") .+ 8272))")
	if !isnan(obj.strength)
		print(io, "($(obj.strength))")
	end
end

"""
	 getPhysSpace(O::AbstractLocalOperator) -> ::VectorSpace

Interface of `AbstractLocalOperator`, return the local physical space.
"""
getPhysSpace(O::IdentityOperator) = O.pspace

"""
	 const tag2Tuple{R₁,R₂} = Tuple{NTuple{R₁,String}, NTuple{R₂,String}}

Type of field `tag` of `LocalOperator`.
"""
const tag2Tuple{R₁, R₂} = Tuple{NTuple{R₁, String}, NTuple{R₂, String}}

"""
	 mutable struct LocalOperator{R₁,R₂} <: AbstractLocalOperator
		  A::AbstractTensorMap
		  name::String
		  si::Int64
		  fermionic::Bool
		  strength::Number
		  tag::tag2Tuple{R₁,R₂}  
	 end

Warpper type for local operators, the building blocks of sparse MPO. 

`R₁` and `R₂` indicate the rank corresponding to codomain and domain, respectively.

Warning: this warpper type does not support automatic converting.

Convention (' marks codomain): 

	 2      2          3         3
	 |      |          |         |
	 A      A--3   1'--A     1'--A--4               
	 |      |          |         |
	 1'     1'         2'        2'
			  
# Constructors
	 LocalOperator(O::AbstractTensorMap,
		  name::Union{String,Symbol},
		  si::Int64,
		  fermionic::Bool,
		  [,strength::Number = NaN]
		  [, tag::tag2Tuple{R₁,R₂}];
		  swap::Bool=false)

Default tag: `"phys"` for physical indices and `name` for virtual indices.  

If `swap == ture`, it will swap the left and right virtual indices.
"""
mutable struct LocalOperator{R₁, R₂} <: AbstractLocalOperator
	A::Union{Nothing, AbstractTensorMap}
	name::String
	si::Int64
	fermionic::Bool
	strength::Number
	tag::tag2Tuple{R₁, R₂}
	function LocalOperator(O::AbstractTensorMap,
		name::String,
		si::Int64,
		fermionic::Bool,
		strength::Number,
		tag::tag2Tuple{R₁, R₂};
		swap::Bool = false) where {R₁, R₂}
		if swap
			perms = (((R₂+2:R₁+R₂)..., R₂), (R₂ + 1, (1:R₂-1)...))
			O = permute(O, perms)
		end
		@assert rank(O, 1) == R₁ && rank(O, 2) == R₂
		return new{R₁, R₂}(O, name, si, fermionic, strength, tag)
	end
	LocalOperator(O::AbstractTensorMap, name::String, si::Int64, fermionic::Bool, tag::tag2Tuple{R₁, R₂}; kwargs...) where {R₁, R₂} = LocalOperator(O, name, si, fermionic, NaN, tag; kwargs...) # default strength = NaN
	function LocalOperator(O::AbstractTensorMap, name::String, si::Int64, fermionic::Bool, strength::Number = NaN; swap::Bool = false)
		# default tag, only for rank ≤ 4
		if swap
			@assert (R₁ = rank(O, 2)) ≤ 2
			@assert (R₂ = rank(O, 1)) ≤ 2
			perms = (((R₂+2:R₁+R₂)..., R₂), (R₂ + 1, (1:R₂-1)...))
			O = permute(O, perms)
		else
			@assert (R₁ = rank(O, 1)) ≤ 2
			@assert (R₂ = rank(O, 2)) ≤ 2
		end
		tag1 = R₁ == 1 ? ("phys",) : ("", "phys")
		tag2 = R₂ == 1 ? ("phys",) : ("phys", "")
		return new{R₁, R₂}(O, name, si, fermionic, strength, (tag1, tag2))
	end
	LocalOperator(O::AbstractTensorMap, name::Symbol, args...; kwargs...) = LocalOperator(O, String(name), args...; kwargs...)
end

hastag(::LocalOperator) = true
getOpName(O::LocalOperator) = O.name
function getPhysSpace(O::LocalOperator)
	return domain(O)[1]
end
isfermionic(O::LocalOperator) = O.fermionic

function Base.show(io::IO, obj::LocalOperator{R₁, R₂}) where {R₁, R₂}
	print(io, "$(obj.name)$(String(collect("$(obj.si)") .+ 8272)){$R₁,$R₂}")
	if !isnan(obj.strength)
		print(io, "($(obj.strength))")
	end
end

"""
	 ==(A::AbstractLocalOperator, B::AbstractLocalOperator) -> ::Bool

Test if two LocalOperator objects are equal. Note we do not consider the field `strength`.
"""
==(::AbstractLocalOperator, ::AbstractLocalOperator) = false
==(A::IdentityOperator, B::IdentityOperator) = (A.si == B.si)
function ==(A::LocalOperator{R₁, R₂}, B::LocalOperator{R₁, R₂}) where {R₁, R₂}
	A.name ≠ B.name && return false
	A.si ≠ B.si && return false
	A.tag ≠ B.tag && return false
	return A.A == B.A
end

"""
	 +(A::LocalOperator{R₁,R₂}, B::LocalOperator{R₁,R₂}) -> ::LocalOperator{R₁,R₂}

Plus of two local operators on the same site. Note the `strength` of each one must not be `NaN`.

Field `name` of output obj is `"A.name(A.strength) + B.name(B.strength)"`.
"""
function +(A::LocalOperator{R₁, R₂}, B::LocalOperator{R₁, R₂}) where {R₁, R₂}
	@assert A.si == B.si && !isnan(A.strength) && !isnan(B.strength)
	@assert A.fermionic == B.fermionic
	Op = A.A * A.strength + B.A * B.strength
	name = "$(A.name)($(A.strength)) + $(B.name)($(B.strength))"
	return LocalOperator(Op, name, A.si, A.fermionic, 1)
end
function +(A::LocalOperator{1, 1}, B::IdentityOperator)
	@assert A.si == B.si && !isnan(A.strength) && !isnan(B.strength)
	@assert !isfermionic(A)
	Op = A.A * A.strength
	add!(Op, id(domain(A)), B.strength)
	name = "$(A.name)($(A.strength)) + I($(B.strength))"
	return LocalOperator(Op, name, A.si, false, 1)
end
+(A::IdentityOperator, B::LocalOperator{1, 1}) = B + A
function +(A::IdentityOperator, B::IdentityOperator)
	@assert A.si == B.si && !isnan(A.strength) && !isnan(B.strength)
	@assert A.pspace == B.pspace
	return IdentityOperator(A.pspace, A.si, A.strength + B.strength)
end

"""
	 *(O::LocalOperator{R₁,R₂}, A::MPSTensor{R₃}) -> ::MPSTensor{R₁ + R₂ +R₃ - 2}

Apply a local operator on a MPS tensor.

Convention:
	   R₁+2  R₁+3 ...
		  | /
	  1---A-- end
		  |
	  2---O --- ...
		/ |
	 ...  R₁+1      
i.e., merge the domain and codomain, legs of `A` first.
"""
function *(O::LocalOperator{R₁, R₂}, A::MPSTensor{R₃}) where {R₁, R₂, R₃}
	pO = (Tuple(setdiff(1:R₁+R₂, R₁ + 1)), (R₁ + 1,))
	pA = ((2,), Tuple(setdiff(1:R₃, 2)))
	pOA = ((R₁ + R₂, 1:R₁...), (R₁+R₂+1:R₁+R₂+R₃-3..., R₁+1:R₁+R₂-1..., R₁ + R₂ + R₃ - 2))

	OA = TensorOperations.tensorcontract(pOA, O.A, pO, :N, A.A, pA, :N, O.strength)
	return MPSTensor(OA)
end

# node we only apply * in addIntr..., hence we assert A.strength == B.strength == NaN 
"""
	 *(A::LocalOperator{R₁, R₂}, B::LocalOperator{R₃,R₄}) -> ::LocalOperator{R₁ + R₃ - 1, R₂ + R₄ - 1}

The multiplication of two local operators. 

Since we only use this function when generating `InteractionTree`, field `strength` of A and B must be `NaN`.

Field `name` of output obj is `"A.name" * "B.name"`.

Warning: we write this function case by case via multiple dispatch, hence it may throw a "no method matching" error for some interactions. 
"""
function *(A::LocalOperator{1, 1}, B::LocalOperator{1, 1})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)
	@tensor O[a; c] := A.A[a b] * B.A[b c]
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic)
end
function *(A::LocalOperator{1, 1}, B::LocalOperator{1, 2})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)
	@tensor O[a; c d] := A.A[a b] * B.A[b c d]
	tag = (A.tag[1], B.tag[2])
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end
function *(A::LocalOperator{1, 1}, B::LocalOperator{2, 1})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)
	@tensor O[c a; d] := A.A[a b] * B.A[c b d]
	tag = ((B.tag[1][1], A.tag[1][1]), B.tag[2])
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end
function *(A::LocalOperator{2, 1}, B::LocalOperator{1, 2})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)
	@tensor O[a b; d e] := A.A[a b c] * B.A[c d e]
	tag = (A.tag[1], B.tag[2])
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end
function *(A::LocalOperator{1, 2}, B::LocalOperator{1, 1})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)
	@tensor O[a; d c] := A.A[a b c] * B.A[b d]
	tag = (A.tag[1], (B.tag[2][1], A.tag[2][2]))
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end
function *(A::LocalOperator{1, 2}, B::LocalOperator{2, 1})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)
	# match tags
	if A.tag[2][2] == B.tag[1][1]
		@tensor O[a; d] := A.A[a b c] * B.A[c b d]
		tag = (A.tag[1], B.tag[2])
	else
		@tensor O[d a; e c] := A.A[a b c] * B.A[d b e]
		tag = ((B.tag[1][1], A.tag[1][1]), (B.tag[2][1], A.tag[2][2]))
	end
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end
function *(A::LocalOperator{1, 2}, B::LocalOperator{1, 2})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)

	@tensor O[a; d c e] := A.A[a b c] * B.A[b d e]
	tag = (A.tag[1], (B.tag[2][1], A.tag[2][2], B.tag[2][2]))
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end
function *(A::LocalOperator{1, 2}, B::LocalOperator{2, 2})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)

	# match tags
	if A.tag[2][2] == B.tag[1][1]
		@tensor O[a; e f] := A.A[a b c] * B.A[c b e f]
		tag = (A.tag[1], B.tag[2])
	else
		@tensor O[d a; e c f] := A.A[a b c] * B.A[d b e f]
		tag = ((B.tag[1][1], A.tag[1][1]), (B.tag[2][1], A.tag[2][2], B.tag[2][2]))
	end
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end
function *(A::LocalOperator{2, 1}, B::LocalOperator{1, 1})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)
	@tensor O[a b; d] := A.A[a b c] * B.A[c d]
	tag = (A.tag[1], B.tag[2])
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end
function *(A::LocalOperator{2, 1}, B::LocalOperator{2, 1})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)

	@tensor O[a d b; e] := A.A[a b c] * B.A[d c e]
	tag = ((A.tag[1][1], B.tag[1][1], A.tag[1][2]), B.tag[2])
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end
function *(A::LocalOperator{2, 2}, B::LocalOperator{2, 1})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)

	# match tags
	if A.tag[2][2] == B.tag[1][1]
		@tensor O[a b; e] := A.A[a b c d] * B.A[d c e]
		tag = (A.tag[1], B.tag[2])
	else
		@tensor O[a e b; f d] := A.A[a b c d] * B.A[e c f]
		tag = ((A.tag[1][1], B.tag[1][1], A.tag[1][2]), (B.tag[2][1], A.tag[2][2]))
	end
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end
function *(A::LocalOperator{2, 2}, B::LocalOperator{2, 2})
	@assert A.si == B.si && isnan(A.strength) && isnan(B.strength)

	# match tags
	if A.tag[2][2] == B.tag[1][1]
		@tensor O[a b; e f] := A.A[a b c d] * B.A[d c e f]
		tag = (A.tag[1], B.tag[2])
	else
		@tensor O[a e b; f d g] := A.A[a b c d] * B.A[e c f g]
		tag = ((A.tag[1][1], B.tag[1][1], A.tag[1][2]), (B.tag[2][1], A.tag[2][2], B.tag[2][2]))
	end
	fermionic = A.fermionic ⊻ B.fermionic
	return LocalOperator(O, A.name * B.name, A.si, fermionic, tag)
end

function _swapOp(obj::LocalOperator{R₁, R₂}) where {R₁, R₂}
	perms = (((R₁+2:R₁+R₂)..., R₁), (R₁ + 1, (1:R₁-1)...))
	O = permute(obj.A, perms)
	tag = ((obj.tag[2][2:end]..., obj.tag[1][end]), (obj.tag[2][1], obj.tag[1][1:end-1]...))
	return LocalOperator(O, obj.name, obj.si, obj.fermionic, obj.strength, tag)
end

function _leftOp(obj::LocalOperator{R₁, R₂}) where {R₁, R₂}
	# transform to a left operator, i.e. R₁ == 1
	perms = ((R₁,), (R₁ + 1, (1:R₁-1)..., (R₁+2:R₁+R₂)...))
	O = permute(obj.A, perms)
	tag = ((obj.tag[1][end],), (obj.tag[2][1], obj.tag[1][1:end-1]..., obj.tag[2][2:end]...))
	return LocalOperator(O, obj.name, obj.si, obj.fermionic, obj.strength, tag)
end
_leftOp(obj::LocalOperator{1, 1}) = obj

function _rightOp(obj::LocalOperator{R₁, R₂}) where {R₁, R₂}
	# transform to a right operator, i.e. R₂ == 1
	perms = (((1:R₁-1)..., (R₁+2:R₁+R₂)..., R₁), (R₁ + 1,))
	O = permute(obj.A, perms)
	tag = ((obj.tag[1][1:end-1]..., obj.tag[2][2:end]..., obj.tag[1][end]), (obj.tag[2][1],))
	return LocalOperator(O, obj.name, obj.si, obj.fermionic, obj.strength, tag)
end
_rightOp(obj::LocalOperator{1, 1}) = obj


# dimension of left/right auxiliary bond
function _vdim(::IdentityOperator, idx::Int64)
	@assert idx == 1 || idx == 2
	return 1, 1
end

function _vdim(::LocalOperator{0, 0}, idx::Int64)
	@assert idx == 1 || idx == 2
	return 1, 1
end
function _vdim(::LocalOperator{1, 1}, idx::Int64)
	@assert idx == 1 || idx == 2
	return 1, 1
end
function _vdim(A::LocalOperator{2, 1}, idx::Int64)
	@assert idx == 1 || idx == 2
	return dim(A.A, 1)
end

function _vdim(A::LocalOperator{1, 2}, idx::Int64)
	@assert idx == 1 || idx == 2
	return dim(A.A, 3)
end

function _vdim(A::LocalOperator{2, 2}, idx::Int64)
	@assert idx == 1 || idx == 2
	return idx == 1 ? dim(A.A, 1) : dim(A.A, 4)
end

