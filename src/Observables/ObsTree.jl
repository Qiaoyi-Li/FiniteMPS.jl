"""
     struct ObservableTree{N, T<:AbstractVector{LocalLeftTensor}, C<: AbstractStoreType}
          Root::InteractionTreeNode{Nothing}
     end
     
Similar to `InteractionTree` but specially used for calculation of observables. 

Value of each non-root node is a length `1` vector of `LocalLeftTensor`, whose concrete type depends on `C <: AbstractStoreType`.

# Constructors
     ObservableTree{N}(;disk::Bool = false)
Initialize an empty object.
"""
struct ObservableTree{N, T<:AbstractVector{LocalLeftTensor}, C<: AbstractStoreType}
     Root::InteractionTreeNode{Nothing}
     function ObservableTree{N}(;disk::Bool = false) where N
          Root = InteractionTreeNode(nothing, nothing)
          for i = 1:N
               value = Vector{LocalLeftTensor}(undef, 1)
               disk && (value = SerializedElementArrays.disk(value))
               addchild!(Root, InteractionTreeNode(IdentityOperator(0), value))
          end
          T = typeof(Root.children[1].value)
          C = disk ? StoreDisk : StoreMemory
          return new{N, T, C}(Root)     
     end
     ObservableTree(;kwargs...) = ObservableTree{1}(;kwargs...)
end

"""
     addObs!(Tree::ObservableTree{M},
          Op::NTuple{N,AbstractTensorMap},
          si::NTuple{N,Int64}
          n::Int64 = 1; 
          kwargs...)

Add a term to the `n`-th root of `ObservableTree{M}`. Detailed usage see `addIntr!`. 
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


     