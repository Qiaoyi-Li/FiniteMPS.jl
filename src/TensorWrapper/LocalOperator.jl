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

"""
     mutable struct IdentityOperator <: AbstractLocalOperator
          si::Int64
          strength::Number 
     end
     
Lazy type of identity operator, used for skipping some tensor contractions.

# Constructors
     IdentityOperator(si::Int64, strength::Number = NaN)
"""
mutable struct IdentityOperator <: AbstractLocalOperator
     si::Int64
     strength::Number 
     function IdentityOperator(si::Int64, strength::Number = NaN)
          return new(si, strength)
     end
end

"""
     getOpName(::AbstractLocalOperator) -> ::String

Interface of `AbstractLocalOperator`, return `"I"` for `IdentityOperator` and `O.name` for `O::LocalOperator`.
"""
getOpName(::IdentityOperator) = "I"

function Base.show(io::IO, obj::IdentityOperator)
     print(io, "I$(String(collect("$(obj.si)") .+ 8272))")
     if !isnan(obj.strength)
          print(io, "($(obj.strength))")
     end
end

"""
     const tag2Tuple{R₁,R₂} = Tuple{NTuple{R₁,String}, NTuple{R₂,String}}

Type of field `tag` of `LocalOperator`.
"""
const tag2Tuple{R₁,R₂} = Tuple{NTuple{R₁,String},NTuple{R₂,String}}

"""
     mutable struct LocalOperator{R₁,R₂} <: AbstractLocalOperator
          A::AbstractTensorMap
          name::String
          si::Int64
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
          [,strength::Number = NaN]
          [, tag::tag2Tuple{R₁,R₂}];
          swap::Bool=false)

Default tag: `"phys"` for physical indices and `name` for virtual indices.  

If `swap == ture`, it will swap the left and right virtual indices.
"""
mutable struct LocalOperator{R₁,R₂} <: AbstractLocalOperator
     A::Union{Nothing,AbstractTensorMap}
     name::String
     si::Int64
     strength::Number 
     tag::tag2Tuple{R₁,R₂}
     function LocalOperator(O::AbstractTensorMap, name::String, si::Int64, strength::Number,
          tag::tag2Tuple{R₁,R₂}; swap::Bool=false) where {R₁,R₂}
          if swap
               perms = (((R₂+2:R₁+R₂)..., R₂), (R₂ + 1, (1:R₂-1)...))
               O = permute(O, perms)
          end
          @assert rank(O, 1) == R₁ && rank(O, 2) == R₂
          return new{R₁,R₂}(O, name, si, strength, tag)
     end
     LocalOperator(O::AbstractTensorMap, name::String, si::Int64, tag::tag2Tuple{R₁,R₂}; kwargs...) where {R₁,R₂} = LocalOperator(O, name, si, NaN, tag; kwargs...) # default strength = NaN
     function LocalOperator(O::AbstractTensorMap, name::String, si::Int64, strength::Number=NaN; swap::Bool=false)
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
          tag1 = R₁ == 1 ? ("phys",) : (name, "phys")
          tag2 = R₂ == 1 ? ("phys",) : ("phys", name)
          return new{R₁,R₂}(O, name, si, strength, (tag1, tag2))
     end
     LocalOperator(O::AbstractTensorMap, name::Symbol, args...; kwargs...) = LocalOperator(O, String(name), args...; kwargs...)
end

hastag(::LocalOperator) = true
getOpName(O::LocalOperator) = O.name

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
     return A.A == B.A
end

"""
     +(A::LocalOperator{R₁,R₂}, B::LocalOperator{R₁,R₂}) -> ::LocalOperator{R₁,R₂}

Plus of two local operators on the same site. Note the `strength` of each one must not be `NaN`.

Field `name` of output obj is `"A.name(A.strength) + B.name(B.strength)"`.
"""
function +(A::LocalOperator{R₁,R₂}, B::LocalOperator{R₁,R₂}) where {R₁,R₂}
     @assert A.si == B.si && !isnan(A.strength) && !isnan(B.strength)
     Op = A.A * A.strength + B.A * B.strength
     name = "$(A.name)($(A.strength)) + $(B.name)($(B.strength))"
     return LocalOperator(Op, name, A.si, 1)
end

# node we only apply * in addIntr..., hence we assert A.strength == B.strength == NaN 
"""
     *(A::LocalOperator{R₁, R₂}, B::LocalOperator{R₃,R₄}) -> ::LocalOperator{R₁ + R₃ - 1, R₂ + R₄ - 1}

The multiplication of two local operators. 

Since we only use this function when generating `InteractionTree`, field `strength` of A and B must be `NaN`.

Field `name` of output obj is `"A.name" * "B.name"`.

Warning: we write this function case by case via multiple dispatch, hence it may throw a "no method matching" error for some interactions. 
"""
function *(A::LocalOperator{1,1}, B::LocalOperator{1,1}) 
     @assert A.si == B.si && isnan(A.strength) && isnan(B.strength)
     @tensor O[a; c] := A.A[a b] * B.A[b c] 
     return LocalOperator(O, A.name * B.name, A.si)
end
function *(A::LocalOperator{1,2}, B::LocalOperator{2,1}) 
     @assert A.si == B.si && isnan(A.strength) && isnan(B.strength)
     @tensor O[a; d] := A.A[a b c] * B.A[c b d] 
     return LocalOperator(O, A.name * B.name, A.si)
end
function *(A::LocalOperator{1,2}, B::LocalOperator{2,2})
     @assert A.si == B.si && isnan(A.strength) && isnan(B.strength)

     # match tags
     if A.tag[2][2] == B.tag[1][1]
          @tensor O[a; e f] := A.A[a b c] * B.A[c b e f]
          tag = (A.tag[1], B.tag[2])
     else
          @tensor O[d a; e c f] := A.A[a b c] * B.A[d b e f]
          tag = ((A.tag[1][1], B.tag[1][1]), (B.tag[2][1], A.tag[2][2], B.tag[2][2]))
     end
     return LocalOperator(O, A.name * B.name, A.si, tag)
end
function *(A::LocalOperator{2,2}, B::LocalOperator{2,1})
     @assert A.si == B.si && isnan(A.strength) && isnan(B.strength)

     # match tags
     if A.tag[2][2] == B.tag[1][1]
          @tensor O[a b; e] := A.A[a b c d] * B.A[d c e]
          tag = (A.tag[1], B.tag[2])
     else
          @tensor O[a e b; f d] := A.A[a b c d] * B.A[e c f]
          tag = ((A.tag[1][1], B.tag[1][1], A.tag[1][2]), (B.tag[2][1], A.tag[2][2]))
     end
     return LocalOperator(O, A.name * B.name, A.si, tag)
end
function *(A::LocalOperator{2,2}, B::LocalOperator{2,2})
     @assert A.si == B.si && isnan(A.strength) && isnan(B.strength)

     # match tags
     if A.tag[2][2] == B.tag[1][1]
          @tensor O[a b; e f] := A.A[a b c d] * B.A[d c e f]
          tag = (A.tag[1], B.tag[2])
     else
          @tensor O[a e b; f d g] := A.A[a b c d] * B.A[e c f g]
          tag = ((A.tag[1][1], B.tag[1][1], A.tag[1][2]), (B.tag[2][1], A.tag[2][2], B.tag[2][2]))
     end
     return LocalOperator(O, A.name * B.name, A.si, tag)
end

function _swapOp(obj::LocalOperator{R₁, R₂}) where {R₁, R₂}
     perms = (((R₁+2:R₁+R₂)..., R₁), (R₁ + 1, (1:R₁-1)...))
     O = permute(obj.A, perms)
     tag = ((obj.tag[2][2:end]..., obj.tag[1][end]), (obj.tag[2][1], obj.tag[1][1:end-1]...))
     return LocalOperator(O, obj.name, obj.si, obj.strength, tag)
end

function _leftOp(obj::LocalOperator{R₁, R₂}) where {R₁, R₂}
     # transform to a left operator, i.e. R₁ == 1
     perms = ((R₁,), (R₁ + 1, (1:R₁-1)..., (R₁ + 2: R₁ + R₂)...))
     O = permute(obj.A, perms)
     tag = ((obj.tag[1][end],), (obj.tag[2][1], obj.tag[1][1:end-1]..., obj.tag[2][2:end]...))
     return LocalOperator(O, obj.name, obj.si, obj.strength, tag)
end


function _rightOp(obj::LocalOperator{R₁, R₂}) where {R₁, R₂}
     # transform to a right operator, i.e. R₂ == 1
     perms = (((1:R₁-1)..., (R₁ + 2: R₁ + R₂)..., R₁), (R₁ + 1,))
     O = permute(obj.A, perms)
     tag = ((obj.tag[1][1:end-1]..., obj.tag[2][2:end]..., obj.tag[1][end]), (obj.tag[2][1], ))
     return LocalOperator(O, obj.name, obj.si, obj.strength, tag)
end

# dimension of left/right auxiliary bond
function _vdim(::IdentityOperator, idx::Int64)
     @assert idx == 1 || idx == 2
     return 1, 1
end

function _vdim(::LocalOperator{0,0}, idx::Int64)
     @assert idx == 1 || idx == 2
     return 1, 1
end
function _vdim(::LocalOperator{1,1}, idx::Int64)
     @assert idx == 1 || idx == 2
     return 1, 1
end
function _vdim(A::LocalOperator{2,1}, idx::Int64)
     @assert idx == 1 || idx == 2
     return dim(A.A, 1)
end

function _vdim(A::LocalOperator{1,2}, idx::Int64)
     @assert idx == 1 || idx == 2
     return dim(A.A, 3)
end

function _vdim(A::LocalOperator{2,2}, idx::Int64)
     @assert idx == 1 || idx == 2
     return idx == 1 ? dim(A.A, 1) : dim(A.A, 4)
end

