"""
     addIntr1!(Root::InteractionTreeNode,
          Op::AbstractTensorMap,
          si::Int64,
          strength::Number;
          name::Union{Symbol,String} = :O) -> nothing

     addIntr1!(Tree::InteractionTree, args...) = addIntr1!(Tree.Root.children[1], args...)

Add an on-site term `Op` at site `si` to a given interaction tree. 

     addIntr1!(Root::InteractionTreeNode, O::LocalOperator, strength::Number) -> nothing

Expert version, each method finally reduces to this one.
"""
function addIntr1!(Root::InteractionTreeNode, Op::AbstractTensorMap, si::Int64, strength::Number;
     name::Union{Symbol,String}=:O)
     # add on-site term

     strength == 0 && return nothing
     O = LocalOperator(Op, name, si)
     return addIntr1!(Root, O, strength)

end
addIntr1!(Tree::InteractionTree, args...; kwargs...) = addIntr1!(Tree.Root.children[1], args...; kwargs...)

function addIntr1!(Root::InteractionTreeNode, O::LocalOperator, strength::Number)
     current_node = Root
     si = 1
     pspace = getPhysSpace(O)
     while si < O.si

          Op_i = IdentityOperator(pspace, si)

          idx = findfirst(x -> x.Op == Op_i, current_node.children)
          if isnothing(idx)
               addchild!(current_node, Op_i)
               current_node = current_node.children[end]
          else
               current_node = current_node.children[idx]
          end
          si += 1
     end

     idx = findfirst(x -> x.Op == O, current_node.children)
     if isnothing(idx)
          addchild!(current_node, O)
          current_node.children[end].Op.strength = strength
     else
          _update_strength!(current_node.children[idx], strength) && deleteat!(current_node.children, idx)
     end

     return nothing
end

