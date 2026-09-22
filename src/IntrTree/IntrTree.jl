"""
	mutable struct InteractionChannel{T}
		Ops::Vector{Int64}
		ref::Ref
		LeafL::T
		LeafR::T
		preserve::Bool
		phase::Number
	end

A concrete type for labeling an interaction channel in an `InteractionTree`.
	
# Fields
	Ops::Vector{Int64}
Stores the indices of local operators.

	ref::Ref 
`Ref` for the interaction strength.
	
	LeafL::T
	LeafR::T
The left and right leaf nodes. 

	preserve::Bool
If `true`, the channel will not be merged so that the interaction strength can be dynamically tuned via `Ref`.

	phase::Number
Output multiplier for observable and ITP channels; defaults to one.

# Constructors
	InteractionChannel(Ops::Vector{Int64},
		ref::Ref,
		LeafL::T,
		LeafR::T,
		preserve::Bool = false; phase::Number = 1) -> InteractionChannel{T}
"""
mutable struct InteractionChannel{T}
	Ops::Vector{Int64}
	ref::Ref
	LeafL::T
	LeafR::T
	preserve::Bool
	phase::Number
	function InteractionChannel(Ops::Vector{Int64}, ref::Ref, LeafL::T, LeafR::T, preserve::Bool = false; phase::Number = 1) where T
		return new{T}(Ops, ref, LeafL, LeafR, preserve, phase)
	end
end
function show(io::IO, obj::InteractionChannel)
	print(io, "InteractionChannel$(obj.Ops)($(obj.ref))")
	return nothing
end

"""
	mutable struct InteractionTreeNode 
		Op::NTuple{2, Int64}
		parent::Union{Nothing, InteractionTreeNode}
		children::Vector{InteractionTreeNode}
		Intrs::Vector{InteractionChannel}
	end

Concrete type for a node of `InteractionTree`.
	
# Fields 
	Op::NTuple{2, Int64}
`(si, idx)` to label the `idx`-st operator at site `si`.

	parent::Union{Nothing, InteractionTreeNode}
	children::Vector{InteractionTreeNode}
The parent node and children nodes.

	Intrs::Vector{InteractionChannel}
Stores all interaction channels linked to this node.

# Constructors
	InteractionTreeNode(Op::NTuple{2, Int64},
		parent::Union{Nothing, InteractionTreeNode},
		children::Vector{InteractionTreeNode} = InteractionTreeNode[],
		Intrs::Vector{InteractionChannel} = InteractionChannel[]
		) -> InteractionTreeNode
"""
mutable struct InteractionTreeNode
	Op::NTuple{2, Int64}
	parent::Union{Nothing, InteractionTreeNode}
	children::Vector{InteractionTreeNode}
	Intrs::Vector{InteractionChannel}
	function InteractionTreeNode(Op::NTuple{2, Int64},
		parent::Union{Nothing, InteractionTreeNode},
		children::Vector{InteractionTreeNode} = InteractionTreeNode[],
		Intrs::Vector{InteractionChannel} = InteractionChannel[])
		return new(Op, parent, children, Intrs)
	end
end
function show(io::IO, obj::InteractionTreeNode)
	print(io, obj.Op[2])
	return nothing
end


parent(node::InteractionTreeNode) = node.parent
children(node::InteractionTreeNode) = node.children
ParentLinks(::Type{InteractionTreeNode}) = StoredParents()
ChildIndexing(::Type{InteractionTreeNode}) = IndexedChildren()
NodeType(::Type{InteractionTreeNode}) = HasNodeType()
nodetype(::Type{InteractionTreeNode}) = InteractionTreeNode

"""
	mutable struct InteractionTree{L}
		Ops::Vector{Vector{AbstractLocalOperator}}
		Refs::Dict{String, Dict}
		RootL::InteractionTreeNode
		RootR::InteractionTreeNode
	end

Implementation of a bi-tree structure for storing all interactions in a `L`-site Hamiltonian.

# Fields
	Ops::Vector{Vector{AbstractLocalOperator}}
Stores all concrete local operators used at each site, `Ops[si][idx]` is the `idx`-st operator at site `si`.

	Refs::Dict{String, Dict}
A dictionary to store all interactions strength. `Refs[name]` is a dictionary `(i, j, ...) => Ref`, where `Ref[]` is the strength of this interaction term with site indices `i, j, ...`.

	RootL::InteractionTreeNode
	RootR::InteractionTreeNode
The root node of left (from the first site) or right (from the last site) tree. 

# Constructors
	InteractionTree(L::Int64) -> ::InteractionTree{L}
Construct an empty `InteractionTree`.
"""
mutable struct InteractionTree{L}
	Ops::Vector{Vector{AbstractLocalOperator}} # Ops[si][idx]
	Refs::Dict{String, Dict}
	RootL::InteractionTreeNode
	RootR::InteractionTreeNode
	function InteractionTree(L::Int64)
		Ops = [AbstractLocalOperator[] for _ in 1:L]
		Refs = Dict{String, Dict}()
		RootL = InteractionTreeNode((0, 0), nothing)
		RootR = InteractionTreeNode((L + 1, 0), nothing)
		return new{L}(Ops, Refs, RootL, RootR)
	end
end
function show(io::IO, obj::InteractionTree{L}) where L
	println(io, typeof(obj), "(")
	for i in 1:L
		print(io, "[")
		for j in 1:length(obj.Ops[i])
			print(io, obj.Ops[i][j])
			j < length(obj.Ops[i]) && print(io, ", ")
		end
		println(io, "]")
	end
	print_tree(io, obj.RootL)
	print_tree(io, obj.RootR)
	print(io, ")")
	return nothing
end

"""
	merge!(Tree::InteractionTree) -> Tree::InteractionTree
Merge interactions with same left or right environment tensor. Symmetrical merging from boundary to bulk is used here, which works well for most cases.
"""
merge!(Tree::InteractionTree) = _merge_paired!(Tree, Val(true))
