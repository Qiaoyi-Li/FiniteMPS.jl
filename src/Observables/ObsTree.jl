"""
     struct ObservableTree{N}
          Root::InteractionTreeNode{Nothing}
     end
     
Similar to `InteractionTree` but specially used for calculation of observables.

# Constructors
     ObservableTree{N}()
Initialize an empty object.
"""
struct ObservableTree{N}
     Root::InteractionTreeNode{Nothing}
     function ObservableTree{N}() where N
          Root = InteractionTreeNode(nothing, nothing)
          for i = 1:N
               addchild!(Root, InteractionTreeNode(IdentityOperator(0), nothing))
          end
          return new{N}(Root)     
     end
     ObservableTree() = ObservableTree{1}()
end

"""
     addObs!(Tree::ObservableTree{M},
          Op::NTuple{N,AbstractTensorMap},
          si::NTuple{N,Int64}
          n::Int64 = 1; 
          Z::Union{Nothing,AbstractTensorMap} = nothing,
          name::NTuple{N,Union{Symbol,String}} = (:A, :B, ...))

Add a term to the `n`-th root of `ObservableTree{M}`. Detailed usage see `addIntr!`. 

Warning: a same `name` can be given to two local operators iff they are exactly the same, otherwise, it will confuse the `convert` function when trying to extract the values stored in the tree to a dictionary. For example, you can simply name `SzSz` correlation as `(:S, :S)`. However this name is inappropriate for `S+S-` correlation. 
"""
function addObs!(Tree::ObservableTree{M},
     Op::NTuple{N,AbstractTensorMap},
     si::NTuple{N,Int64},
     n::Int64 = 1; 
     kwargs...) where {N, M}
     @assert n ≤ M
     return addIntr!(Tree.Root.children[n], Op, si, 1; kwargs...)
end
function addObs!(Tree::ObservableTree,
     Op::AbstractTensorMap,
     si::Int64,
     n::Int64 = 1; 
     kwargs...)
     return addIntr!(Tree.Root.children[n], Op, si, 1; kwargs...)
end


     