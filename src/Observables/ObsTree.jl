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

# """
# 	 addObs!(Tree::ObservableTree,
# 		  Op::NTuple{N,AbstractTensorMap},
# 		  si::NTuple{N,Int64},
# 		  fermionic::NTuple{N,Bool},
# 		  n::Int64 = 1; 
# 		  kwargs...)

# Add a term to the `n`-th root of `ObservableTree`. This function is a wrapper of `addIntr!` with `Obs = true`. Detailed usage please refer to `addIntr!`. 
# """
# function addObs!(Tree::ObservableTree{M},
# 	Op::NTuple{N, AbstractTensorMap},
# 	si::NTuple{N, Int64},
# 	fermionic::NTuple{N, Bool},
# 	n::Int64 = 1;
# 	kwargs...) where {N, M}
# 	@assert n ≤ M
# 	return addIntr!(Tree.Root.children[n], Op, si, fermionic, 1; Obs = true, kwargs...)
# end
# function addObs!(Tree::ObservableTree,
# 	Op::AbstractTensorMap,
# 	si::Int64,
# 	n::Int64 = 1;
# 	kwargs...)
# 	return addIntr!(Tree.Root.children[n], Op, si, 1; Obs = true, kwargs...)
# end


