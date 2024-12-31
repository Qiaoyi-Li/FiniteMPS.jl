"""
	 AutomataMPO(Tree::InteractionTree,
		  L::Int64 = treeheight(Tree.Root) - 1)
		  -> ::SparseMPO

Convert an interaction tree to a sparse MPO.
"""
function AutomataMPO(Tree::InteractionTree, L::Int64 = treeheight(Tree.Root) - 1)
	# convert an interaction tree to a sparse MPO
	Root = Tree.Root

	# count_size
	D = zeros(Int64, L + 1) # D[i] denotes the bond dimension from i-1 to i
	for node in PreOrderDFS(Root)
		isnothing(node.Op) && continue
		isempty(node.children) && continue
		D[node.Op.si+1] += 1
	end

	# additional channel to store the accumulation
	D[2:end] .+= 1

	H = Vector{SparseMPOTensor}(undef, L)
	for i ∈ 1:L
		H[i] = SparseMPOTensor(nothing, D[i], D[i+1])
	end

	c = vcat(0, repeat([1], L))
	for node in PreOrderDFS(Root)
		isnothing(node.Op) && continue
		si = node.Op.si
		if si == 0
			c[1] += 1
			continue
		end

		# merge this channel to accumulation
		if !isnan(node.Op.strength)
			H[si][c[si], 1] += deepcopy(node.Op)
		end

		# propagate
		if !isempty(node.children)
			c[si+1] += 1
			H[si][c[si], c[si+1]] = _convertStrength(node.Op)
		end

	end

	# remember the identity to propagate accumulation
	for si ∈ 2:L
		idx = findfirst(!isnothing, H[si])
		pspace = isnothing(idx) ? nothing : getPhysSpace(H[si][idx])
		H[si][1, 1] = IdentityOperator(pspace, si, 1)
	end

	# right-to-left merge
	for si in reverse(2:L)
		i = 2
		while i < size(H[si], 1)
			j = findfirst(i+1:size(H[si], 1)) do j
				# find same row
				any(k -> H[si][i, k] != H[si][j, k], 1:size(H[si], 2)) && return false
				lscoef = []
				for k in 1:size(H[si], 2)
					if !isnothing(H[si][i, k])
						push!(lscoef, H[si][i, k].strength / H[si][j, k].strength)
					end
				end
				return all(c -> c == lscoef[1], lscoef)
			end
			if !isnothing(j)
				j += i
				strength_i = H[si][i, findfirst(!isnothing, H[si][i, :])].strength
				strength_j = H[si][j, findfirst(!isnothing, H[si][j, :])].strength
				for idx in 1:size(H[si-1], 1)
					if !isnothing(H[si-1][idx, i])
						H[si-1][idx, i].strength *= strength_i
					end
					if !isnothing(H[si-1][idx, j])
						H[si-1][idx, j].strength *= strength_j
					end
				end
				for idx in 1:size(H[si], 2)
					if !isnothing(H[si][i, idx])
						H[si][i, idx].strength = 1.0
					end
				end

				# merge 
				for idx in 1:size(H[si-1], 1)
					H[si-1][idx, i] += H[si-1][idx, j]
				end
				# remove j 
				H[si-1] = H[si-1][:, vcat(1:j-1, j+1:size(H[si-1], 2))]
				H[si] = H[si][vcat(1:j-1, j+1:size(H[si], 1)), :]

			else
				i += 1
			end

		end
	end

	return SparseMPO(H)
end
AutomataMPO(Root::InteractionTreeNode, args...; kwargs...) = AutomataMPO(InteractionTree(Root), args...; kwargs...)

function _convertStrength(A::AbstractLocalOperator)
	B = deepcopy(A)
	B.strength = 1
	return B
end
