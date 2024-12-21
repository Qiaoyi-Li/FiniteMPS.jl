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

function _update_strength!(node::InteractionTreeNode{T}, strength::Number) where T <: Tuple{String, Vararg{Int64}}
     # directly update the value when used for calculating observables
     node.Op.strength = strength
     return false
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
_swap(A::LocalOperator{2, 1}, B::LocalOperator{1, 1}) = B, A
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






