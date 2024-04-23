"""
     struct ObservableTree{N}
          Root::InteractionTreeNode{Tuple{String, Vararg{Int64}}}
     end
     
Similar to `InteractionTree` but specially used for calculation of observables. The `value` field is used to tell which observable corresponds to the current node.

# Constructors
     ObservableTree{N}()
Initialize an empty object.
"""
struct ObservableTree{N}
     Root::InteractionTreeNode{Tuple{String, Vararg{Int64}}}
     function ObservableTree{N}() where N
          Root = InteractionTreeNode{Tuple{String, Vararg{Int64}}}(nothing, ("",), nothing)
          for i = 1:N
               addchild!(Root, InteractionTreeNode{Tuple{String, Vararg{Int64}}}(IdentityOperator(0),("",),nothing))
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

Warning: there is a known issue that only one permutation will be calculated if there exist multiple permutations with the same operator and name. For example, if you add `(:Sz,:Sz)` with sites `(1, 2)` and `(2, 1)`, only one of them will appear in the final result. Changing the name to distinguish the 1st and 2nd `Sz` by using e.g. `(:Sz1, :Sz2)` can solve this issue. However, a more elegant solution is to avoid adding both of them as we know the expected value must be the same. Thus, we will not prioritize fixing this issue in the near future.
"""
function addObs!(Tree::ObservableTree{M},
     Op::NTuple{N,AbstractTensorMap},
     si::NTuple{N,Int64},
     n::Int64 = 1; 
     kwargs...) where {N, M}
     @assert n ≤ M
     return addIntr!(Tree.Root.children[n], Op, si, 1; Obs = true, kwargs...)
end
function addObs!(Tree::ObservableTree,
     Op::AbstractTensorMap,
     si::Int64,
     n::Int64 = 1; 
     kwargs...)
     return addIntr!(Tree.Root.children[n], Op, si, 1; Obs = true, kwargs...)
end


     