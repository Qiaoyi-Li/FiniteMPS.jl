"""
     addIntr2!(Root::InteractionTreeNode,
          Op::NTuple{2,AbstractTensorMap},
          si::NTuple{2,Int64},
          strength::Number;
          Z::Union{Nothing,AbstractTensorMap} = nothing,
          name::NTuple{2,Union{Symbol,String}} = (:A, :B)) -> nothing

     addIntr2!(Tree::InteractionTree, args...) = addIntr2!(Tree.Root.children[1], args...)

Add a two-site interaction `Op` at site `si` (2tuple) to a given interaction tree. If Z is given, assume Op is ferminic operator and add Z automatically.

     addIntr2!(Root::InteractionTreeNode,
          OL::LocalOperator,
          OR::LocalOperator,
          strength::Number,
          Z::Union{Nothing,AbstractTensorMap}) -> nothing

Expert version, each method finally reduces to this one. 

Note if `OL.si == OR.si`, it will recurse to `addIntr1!` automatically.
"""
function addIntr2!(Root::InteractionTreeNode, Op::NTuple{2,AbstractTensorMap}, si::NTuple{2,Int64}, strength::Number;
     Z::Union{Nothing,AbstractTensorMap}=nothing,
     name::NTuple{2,Union{Symbol,String}}=(:A, :B))
     # add two site interaction term

     strength == 0 && return nothing

     if si[1] == si[2]
          OL = LocalOperator(Op[1], name[1], si[1])
          OR = LocalOperator(Op[2], name[2], si[2])
          return addIntr1!(Root, OL*OR, strength)
     end

     if si[1] < si[2]
          OL = LocalOperator(Op[1], name[1], si[1])
          OR = LocalOperator(Op[2], name[2], si[2])
          !isnothing(Z) && _addZ!(OR, Z)
     else
          OL = LocalOperator(Op[2], name[2], si[2]; swap = true)
          OR = LocalOperator(Op[1], name[1], si[1]; swap = true)
          if !isnothing(Z)
               _addZ!(OR, Z)
               strength = - strength # note ZFZ = -F
          end
     end
     return addIntr2!(Root, OL, OR, strength, Z)

end
addIntr2!(Tree::InteractionTree, args...) = addIntr2!(Tree.Root.children[1], args...)

function addIntr2!(Root::InteractionTreeNode, OL::LocalOperator, OR::LocalOperator,
     strength::Number, Z::Union{Nothing,AbstractTensorMap})
     @assert OL.si < OR.si

     current_node = Root
     si = 1
     pspace = getPhysSpace(OL)
     while si < OR.si

          if si == OL.si
               Op_i = OL
          elseif !isnothing(Z) && OL.si < si < OR.si
               Op_i = LocalOperator(Z, :Z, si)
          else
               Op_i = IdentityOperator(pspace, si)
          end

          idx = findfirst(x -> x.Op == Op_i, current_node.children)
          if isnothing(idx)
               addchild!(current_node, Op_i)
               current_node = current_node.children[end]
          else
               current_node = current_node.children[idx]
          end
          si += 1
     end

     idx = findfirst(x -> x.Op == OR, current_node.children)
     if isnothing(idx)
          addchild!(current_node, OR)
          current_node.children[end].Op.strength = strength
     else
          _update_strength!(current_node.children[idx], strength) && deleteat!(current_node.children, idx)
     end

     return nothing

end

