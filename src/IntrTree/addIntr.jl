"""
     addIntr!(Root::InteractionTreeNode,
          Op::NTuple{N,AbstractTensorMap},
          si::NTuple{N,Int64},
          strength::Number;
          kwargs...) 

Generic function to add an N-site interaction via `addIntr1!`, `addIntr2!` and `addIntr4!`.

# Kwargs
     Z::Union{Nothing,AbstractTensorMap}=nothing
     name::NTuple{N,Union{Symbol,String}}

Detailed usage of kwargs see `addIntr1!`, `addIntr2!` and `addIntr4!`.

     Obs::Bool = false
`Obs == true` means this interaction is used for calculating observables, and thus the `name` and `si` information will be stored in the last node additionally.   
"""
function addIntr!(Root::InteractionTreeNode,
     Op::NTuple{1,AbstractTensorMap},
     si::NTuple{1,Int64},
     strength::Number;
     kwargs...)
     return addIntr1!(Root, Op[1], si[1], strength; kwargs...)
end
function addIntr!(Root::InteractionTreeNode,
     Op::AbstractTensorMap,
     si::Int64,
     strength::Number;
     kwargs...)
     return addIntr1!(Root, Op, si, strength; kwargs...)
end
function addIntr!(Root::InteractionTreeNode,
     Op::NTuple{2,AbstractTensorMap},
     si::NTuple{2,Int64},
     strength::Number;
     kwargs...)
     return addIntr2!(Root, Op, si, strength; kwargs...)
end
function addIntr!(Root::InteractionTreeNode,
    Op::NTuple{3,AbstractTensorMap},
    si::NTuple{3,Int64},
    strength::Number;
    kwargs...)
    return addIntr3!(Root, Op, si, strength; kwargs...)
end
function addIntr!(Root::InteractionTreeNode,
     Op::NTuple{4,AbstractTensorMap},
     si::NTuple{4,Int64},
     strength::Number;
     kwargs...)
     return addIntr4!(Root, Op, si, strength; kwargs...)
end

function _update_strength!(node::InteractionTreeNode, strength::Number)
     if isnan(node.Op.strength)
          node.Op.strength = strength
     else
          node.Op.strength += strength
     end
     # delete this channel if strength == 0
     if node.Op.strength == 0
          # delete
          isempty(node.children) && return true # delete or not
          # propagate
          node.Op.strength = NaN 
     end
     return false
end
function _update_strength!(node::InteractionTreeNode{T}, strength::Number) where T <: Dict{<:Tuple, String}
     # used for calculating observables
     if isnan(node.Op.strength)
          node.Op.strength = strength
     else
          @assert node.Op.strength == strength
     end
     return false
end

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
     OR.A = - OR.A * Z
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

# swap two operators to deal with horizontal bond
_swap(A::LocalOperator{1, 1}, B::LocalOperator{1, 1}) = B, A
_swap(A::LocalOperator{1, 1}, B::LocalOperator{1, 2}) = B, A
_swap(A::LocalOperator{1, 2}, B::LocalOperator{1, 1}) = B, A
_swap(A::LocalOperator{2, 1}, B::LocalOperator{1, 1}) = B, A
_swap(A::LocalOperator{1, 1}, B::LocalOperator{2, 1}) = B, A
function _swap(A::LocalOperator{1, 2}, B::LocalOperator{2, 1})
     return _swapOp(B), _swapOp(A)
end
function _swap(A::LocalOperator{1, 2}, B::LocalOperator{2, 2})
     #  |      |          |      |
     #  A--  --B--va -->  B--  --A--va
     #  |      |          |      |

     @tensor AB[d e; a b f] := A.A[a b c] * B.A[c d e f]     
     # QR 
     TA, TB = leftorth(AB)
     
     return LocalOperator(permute(TA, (1,), (2, 3)), B.name, B.si, B.strength), LocalOperator(permute(TB, (1, 2), (3, 4)), A.name, A.si, A.strength)
end
function _swap(A::LocalOperator{2, 2}, B::LocalOperator{2,1})
     #     |     |         |     |
     # va--A-- --B --> va--B-- --A 
     #     |     |         |     |

     @tensor AB[a e f; b c] := A.A[a b c d] * B.A[d e f]
     # QR
     TA, TB = rightorth(AB)

     return LocalOperator(permute(TA, (1, 2), (3, 4)), B.name, B.si, B.strength), LocalOperator(permute(TB, (1, 2), (3,)), A.name, A.si, A.strength)
end

# fuse the possible multiple bonds between two operators
function _reduceOp(A::LocalOperator{1, R}, B::LocalOperator{R, 1}) where R 
     # match tags 
     tagA = A.tag[2][2:end]
     tagB = B.tag[1][1:end-1]
     permB = map(tagA) do t
          findfirst(x -> x == t, tagB)
     end 
     pA = ((1, 2), Tuple(3:R+1))
     pB = (permB, (R, R+1))
     pC = ((1,2), (3, 4))
     AB = tensorcontract(pC, A.A, pA, :N, B.A, pB, :N)
     # QR 
     TA, TB = leftorth(AB)
     if !isnan(A.strength) && !isnan(B.strength)
          sA = 1
          sB = A.strength * B.strength
     else
          sA = sB = NaN
     end
     OA = LocalOperator(permute(TA, (1,), (2, 3)), A.name, A.si, sA)
     OB = LocalOperator(permute(TB, (1, 2), (3,)), B.name, B.si, sB)
     return OA, OB
end
_reduceOp(A::LocalOperator{1, 1}, B::LocalOperator{1, 1}) = A, B
_reduceOp(A::LocalOperator{1, 2}, B::LocalOperator{2, 1}) = A, B
# contract the possible outer bonds between 3 operators
# C AD B case from 4-site terms
#   |     |     |
# --A-- --B-- --C--   
#   |     |     |
function _reduceOp(A::LocalOperator{2, 2}, B::LocalOperator{2,2}, C::LocalOperator{2,2})
     @tensor ABC[b c; e f h i] := A.A[a b c d] * B.A[d e f g] * C.A[g h i a]
     # QR
     TA, TBC = leftorth(ABC)
     TB, TC = leftorth(permute(TBC, (1, 2, 3), (4, 5)))
     # strength
     if isnan(A.strength) || isnan(B.strength) || isnan(C.strength)
          sA = sB = sC = NaN
     else
          sA = sB = 1
          sC = A.strength * B.strength * C.strength
     end
     OA = LocalOperator(permute(TA, (1,), (2, 3)), A.name, A.si, sA)
     OB = LocalOperator(permute(TB, (1, 2), (3, 4)), B.name, B.si, sB)
     OC = LocalOperator(permute(TC, (1, 2), (3,)), C.name, C.si, sC)
     return OA, OB, OC
end
_reduceOp(A::LocalOperator{1, 2}, B::LocalOperator{2, 2}, C::LocalOperator{2, 1}) = A, B, C
_reduceOp(A::LocalOperator{1, 1}, B::LocalOperator{1, 1}, C::LocalOperator{1, 1}) = A, B, C