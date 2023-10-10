"""
     calObs!(Tree::ObservableTree, Ψ::AbstractMPS{L}; kwargs...) -> Tree::InteractionTree

Calculate observables respect to state `Ψ`, the info to tell which observables to calculate is stored in `Tree`. The results are stored in each leave node of `Tree`. 
"""
function calObs!(Tree::ObservableTree, Ψ::AbstractMPS{L}; kwargs...) where L
     if get(kwargs, :distributed, false) && nworkers() > 1

          tasks = map(eachindex(Tree.Root.children)) do i
               let Root = Tree.Root.children[i]
                    @spawnat :any calObs!(Root, Ψ; kwargs...)
               end
          end

          for i in eachindex(tasks)
               Tree.Root.children[i] = fetch(tasks[i])
          end

     else
          @threads for child in Tree.Root.children
               calObs!(child, Ψ; kwargs...)
          end
     end
     return Tree
end


function calObs!(Root::InteractionTreeNode{T}, Ψ::AbstractMPS{L}; kwargs...) where {L, T<:AbstractVector{LocalLeftTensor}}

     @assert Center(Ψ)[2] ≤ 1 # note right-canonical form is used 

     # initial El
     Root.value[1] = get(kwargs, :El, isometry(codomain(Ψ[1])[1], codomain(Ψ[1])[1]))

     function _calObs_layer!(node::InteractionTreeNode{T}; kwargs...)

          si = node.Op.si

          # current node
          if isnan(node.Op.strength)
               # propagate
               node.Op.strength = 1
               node.value[1] = _pushright(AbstractTrees.parent(node).value[1], Ψ[si]', node.Op, Ψ[si])
               node.Op.strength = NaN
          else
               fac = node.Op.strength
               @assert abs(fac) == 1 # just convention, -1 for some fermionic cases
               node.Op.strength = 1 # we do not want to propagate -El !!
               node.value[1] = _pushright(AbstractTrees.parent(node).value[1], Ψ[si]', node.Op, Ψ[si])
               node.Op.strength = fac * tr(node.value[1].A) # add fac(±1) here
          end

          if !isempty(node.children)
               @threads for child in node.children
                    _calObs_layer!(child; kwargs...)
               end
          end

     end

     @threads for child in Root.children
          _calObs_layer!(child; kwargs...)
     end

     GC.gc()

     return Root
end



