"""
	struct ObservableTree{L}
		Ops::Vector{Vector{AbstractLocalOperator}}
		Refs::Dict{String, Dict}
		RootL::InteractionTreeNode
		RootR::InteractionTreeNode
	end
	 
Similar to `InteractionTree` but specially used for calculation of observables.

# Constructors
	 ObservableTree(L) 
Initialize an empty object, where `L` is the number of sites.
"""
mutable struct ObservableTree{L}
	Ops::Vector{Vector{AbstractLocalOperator}} # Ops[si][idx]
	Refs::Dict{String, Dict}
	RootL::InteractionTreeNode
	RootR::InteractionTreeNode
	function ObservableTree(L::Int64)
		Ops = [AbstractLocalOperator[] for _ in 1:L]
		Refs = Dict{String, Dict}()
		RootL = InteractionTreeNode((0, 0), nothing)
		RootR = InteractionTreeNode((L + 1, 0), nothing)
		return new{L}(Ops, Refs, RootL, RootR)
	end
end

function show(io::IO, obj::ObservableTree{L}) where L
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
	merge!(Tree::ObservableTree) -> Tree::ObservableTree

Merge observables in `Tree` that share the same parent node to reduce the computation cost of environment tensors. The behavior is similar to `merge!` for `InteractionTree`, but some detailed logic is different.
"""
merge!(Tree::ObservableTree) = _merge_paired!(Tree, Val(false))

"""
	treewidth(Tree::ObservableTree) -> Tuple{Int64, Int64}

Return the tree width of the left and right trees. 
"""
function treewidth(Tree::ObservableTree)
	return map([Tree.RootL, Tree.RootR]) do R
		si_last = 0
		n = 1
		width = 1
		for node in StatelessBFS(R)
			si = node.Op[1]
			if si != si_last
				width = max(width, n)
				n = 1
			else
				n += 1
			end
			si_last = si
		end
		return max(width, n)
	end |> Tuple
end
