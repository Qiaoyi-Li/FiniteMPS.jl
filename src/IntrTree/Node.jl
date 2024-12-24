"""
	 mutable struct InteractionTreeNode{T} 
		  Op::Union{Nothing, AbstractLocalOperator}
		  value::Union{Nothing, T}
		  parent::Union{Nothing,InteractionTreeNode}
		  children::Vector{InteractionTreeNode}
	 end

Type of node of interaction tree, with field `Op` to store the local operator, `value` to store anything others.
 
# Constructors
	 InteractionTreeNode(Op::Union{Nothing,AbstractLocalOperator},
		  value::T,
		  parent::Union{Nothing,InteractionTreeNode} = nothing,
		  children::Vector{InteractionTreeNode} = []) -> ::InteractionTreeNode{T}
"""
mutable struct InteractionTreeNode{T}
	Op::Union{Nothing, AbstractLocalOperator}
	value::Union{Nothing, T}
	parent::Union{Nothing, InteractionTreeNode}
	children::Vector{InteractionTreeNode}
	function InteractionTreeNode{T}(Op::Union{Nothing, AbstractLocalOperator},
		value::T,
		parent::Union{Nothing, InteractionTreeNode} = nothing,
		children::Vector{InteractionTreeNode} = InteractionTreeNode[]) where T
		return new{T}(Op, value, parent, children)
	end
	function InteractionTreeNode(Op::Union{Nothing, AbstractLocalOperator},
		value::T,
		parent::Union{Nothing, InteractionTreeNode},
		children::Vector{InteractionTreeNode} = InteractionTreeNode[]) where T
		return new{T}(Op, value, parent, children)
	end
	function InteractionTreeNode(Op::Union{Nothing, AbstractLocalOperator},
		value::T,
		children::Vector{InteractionTreeNode} = InteractionTreeNode[]) where T
		return InteractionTreeNode(Op, value, nothing, children)
	end
	InteractionTreeNode() = InteractionTreeNode(IdentityOperator(0), nothing)
end

function Base.show(io::IO, Root::InteractionTreeNode)
	print_tree(io, Root; maxdepth = 16)
end

AbstractTrees.nodevalue(node::InteractionTreeNode) = node.Op
AbstractTrees.parent(node::InteractionTreeNode) = node.parent
AbstractTrees.children(node::InteractionTreeNode) = node.children
AbstractTrees.ParentLinks(::Type{InteractionTreeNode}) = StoredParents()
AbstractTrees.ChildIndexing(::Type{InteractionTreeNode}) = IndexedChildren()
AbstractTrees.NodeType(::Type{InteractionTreeNode}) = HasNodeType()
AbstractTrees.nodetype(::Type{InteractionTreeNode}) = InteractionTreeNode

"""
	 addchild!(node::InteractionTreeNode, child::InteractionTreeNode) -> nothing

Add a `child` to a given `node`.

	 addchild!(node::InteractionTreeNode, Op::AbstractLocalOperator [, value]) -> nothing

Initialize a child node with `Op`, and add it to the given `node`.

Note we will use `similar`(for array-like types) or `zero`(for other types) to initialize the field `value` in default case, which may lead to a "no method matching" error.
"""
function addchild!(node::InteractionTreeNode, child::InteractionTreeNode)
	isnothing(child.parent) ? child.parent = node : @assert child.parent == node
	push!(node.children, child)
	return nothing
end
function addchild!(node::InteractionTreeNode{Nothing}, Op::AbstractLocalOperator)
	return addchild!(node, Op, nothing)
end
function addchild!(node::InteractionTreeNode{T}, Op::AbstractLocalOperator) where T <: AbstractArray
	return addchild!(node, Op, similar(node.value))
end
function addchild!(node::InteractionTreeNode{T}, Op::AbstractLocalOperator) where T <: Number
	return addchild!(node, Op, zero(node.value))
end
function addchild!(node::InteractionTreeNode, Op::AbstractLocalOperator, value)
	child = InteractionTreeNode(Op, value, node)
	return addchild!(node, child)
end
# for Obs tree
function addchild!(node::InteractionTreeNode{T}, Op::AbstractLocalOperator, ::Nothing) where T <: Dict{Tuple, String}
	# convert nothing to the required type 
	return addchild!(node, Op, Dict{Tuple, String}())
end
function addchild!(node::InteractionTreeNode{T}, Op::AbstractLocalOperator, value::Dict{Tuple, String} = Dict{Tuple, String}()) where T <: Dict{Tuple, String}
	child = InteractionTreeNode{Dict{Tuple, String}}(Op, value, node)
	return addchild!(node, child)
end
function addchild!(node::InteractionTreeNode{T}, Op::AbstractLocalOperator, value::Pair{<:Tuple, String}) where T <: Dict{Tuple, String}
	child = InteractionTreeNode{Dict{Tuple, String}}(Op, Dict{Tuple, String}(value), node)
	return addchild!(node, child)
end



"""
	 struct InteractionTree{N, T}
		  Root::InteractionTreeNode{Nothing}
	 end

Wrapper type of the root node in interaction tree. 

`N` is the children number of root node. When generating sparse MPO, `N` is exactly the length of boundary left environment tensor. Usually `N == 1`, except for some cases when we want to deal with several sparse MPOs independently, e.g. H and N in tanTRG when using fixed particle number technique.

`T` is the type of value of all non-root nodes. Note value of `Root` is always of type `Nothing`. 
"""
struct InteractionTree{N, T}
	Root::InteractionTreeNode{Nothing}
	function InteractionTree(child::InteractionTreeNode{T}...) where T
		N = length(child)

		Root = InteractionTreeNode(nothing, nothing)
		for i in child
			addchild!(Root, i)
		end
		return new{N, T}(Root)
	end
end

# Base.show(io::IO, Tree::InteractionTree) = show(io, Tree.Root)

