"""
     struct ObservableTree{N}
          Root::InteractionTreeNode{Dict{Tuple, String}}
     end
     
Similar to `InteractionTree` but specially used for calculation of observables. The `value` field is used to tell which observable corresponds to the current node. `N` is used to divide the tree into `N` parts. In most cases, the default `N = 1` is sufficient.

# Constructors
     ObservableTree{N}() 
Initialize an empty object. Default `N=1`.
"""
struct ObservableTree{N}
     Root::InteractionTreeNode{Dict{Tuple, String}}
     function ObservableTree{N}() where N
          Root = InteractionTreeNode{Dict{Tuple, String}}(nothing, Dict{Tuple, String}(), nothing)
          for i = 1:N
               addchild!(Root, InteractionTreeNode{Dict{Tuple, String}}(IdentityOperator(0), Dict{Tuple, String}(),nothing))
          end
          return new{N}(Root)     
     end
     ObservableTree() = ObservableTree{1}()
end

"""
     addObs!(Tree::ObservableTree,
          Op::NTuple{N,AbstractTensorMap},
          si::NTuple{N,Int64},
          fermionic::NTuple{N,Bool},
          n::Int64 = 1; 
          kwargs...)

Add a term to the `n`-th root of `ObservableTree`. This function is a wrapper of `addIntr!` with `Obs = true`. Detailed usage please refer to `addIntr!`. 
"""
function addObs!(Tree::ObservableTree{M},
     Op::NTuple{N,AbstractTensorMap},
     si::NTuple{N,Int64},
     fermionic::NTuple{N,Bool},
     n::Int64 = 1; 
     kwargs...) where {N, M}
     @assert n ≤ M
     return addIntr!(Tree.Root.children[n], Op, si, fermionic, 1; Obs = true, kwargs...)
end
function addObs!(Tree::ObservableTree,
     Op::AbstractTensorMap,
     si::Int64,
     n::Int64 = 1; 
     kwargs...)
     return addIntr!(Tree.Root.children[n], Op, si, 1; Obs = true, kwargs...)
end


     