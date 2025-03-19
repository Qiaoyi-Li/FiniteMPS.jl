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

function merge!(Tree::ObservableTree{L}) where L

	lsnode_L = InteractionTreeNode[Tree.RootL]
	lschild_L = InteractionTreeNode[]
	lsnode_R = InteractionTreeNode[Tree.RootR]
	lschild_R = InteractionTreeNode[]
	for si_count in 0:L
		# left to right
		si = si_count
		empty!(lschild_L)
		for node in lsnode_L
			if length(node.Intrs) < 2
				append!(lschild_L, node.children)
				continue
			end

			# search 
			i = 1
			while i < length(node.Intrs)
				ch_i = node.Intrs[i]
				if length(ch_i.Ops) == 0
					# do not merge equivalent terms in obs mode
					i += 1
				else
					ids = findall(1:length(node.Intrs)) do j
						j < i && return false
						ch_j = node.Intrs[j]
						length(ch_j.Ops) == 0 && return false
						return ch_j.Ops[1] == ch_i.Ops[1]
					end
					if length(ids) < 2
						i += 1
						continue
					end

					# add child 
					node_new = InteractionTreeNode((si + 1, ch_i.Ops[1]), node)
					push!(node.children, node_new)
					for j in ids
						ch_j = node.Intrs[j]
						# update ch_j 
						deleteat!(ch_j.Ops, 1)
						ch_j.LeafL = node_new

						push!(node_new.Intrs, ch_j)
					end

					deleteat!(node.Intrs, ids)

				end

			end

			append!(lschild_L, node.children)
		end

		# next layer
		empty!(lsnode_L)
		append!(lsnode_L, lschild_L)

		# right to left
		si = L - si_count + 1
		empty!(lschild_R)
		for node in lsnode_R
			if length(node.Intrs) < 2
				append!(lschild_R, node.children)
				continue
			end

			# search
			i = 1
			while i < length(node.Intrs)
				ch_i = node.Intrs[i]
				if length(ch_i.Ops) == 0
					# do not merge equivalent terms in obs mode
					i += 1

				else
					ids = findall(1:length(node.Intrs)) do j
						j < i && return false
						ch_j = node.Intrs[j]
						length(ch_j.Ops) == 0 && return false
						return ch_j.Ops[end] == ch_i.Ops[end]
					end
					if length(ids) < 2
						i += 1
						continue
					end

					# add child
					node_new = InteractionTreeNode((si - 1, ch_i.Ops[end]), node)
					push!(node.children, node_new)
					for j in ids
						ch_j = node.Intrs[j]
						# update ch_j
						deleteat!(ch_j.Ops, length(ch_j.Ops))
						ch_j.LeafR = node_new

						push!(node_new.Intrs, ch_j)
					end

					deleteat!(node.Intrs, ids)

				end

			end

			append!(lschild_R, node.children)
		end

		# next layer
		empty!(lsnode_R)
		append!(lsnode_R, lschild_R)
	end

	# add nodes  
	lsnode_L = InteractionTreeNode[Tree.RootL]
	lschild_L = InteractionTreeNode[]
	lsnode_R = InteractionTreeNode[Tree.RootR]
	lschild_R = InteractionTreeNode[]
	for si_count in 0:L
		# left to right
		si = si_count
		empty!(lschild_L)
		for node in lsnode_L
			ids = findall(node.Intrs) do ch
				length(ch.Ops) ≥ 1
			end

			for i in ids
				ch = node.Intrs[i]
				node_new = InteractionTreeNode((si + 1, ch.Ops[1]), node)
				push!(node.children, node_new)

				# update ch
				deleteat!(ch.Ops, 1)
				ch.LeafL = node_new

				push!(node_new.Intrs, ch)
			end
			deleteat!(node.Intrs, ids)

			append!(lschild_L, node.children)
		end

		append!(empty!(lsnode_L), lschild_L)

		# right to left
		si = L - si_count + 1
		empty!(lschild_R)
		for node in lsnode_R
			ids = findall(node.Intrs) do ch
				length(ch.Ops) ≥ 1
			end

			for i in ids
				ch = node.Intrs[i]
				node_new = InteractionTreeNode((si - 1, ch.Ops[end]), node)
				push!(node.children, node_new)

				# update ch
				deleteat!(ch.Ops, length(ch.Ops))
				ch.LeafR = node_new

				push!(node_new.Intrs, ch)
			end
			deleteat!(node.Intrs, ids)

			append!(lschild_R, node.children)
		end

		append!(empty!(lsnode_R), lschild_R)

	end

	return Tree
end

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