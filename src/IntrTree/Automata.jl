function AutomataMPO(Tree::InteractionTree{L}; tol::Float64 = 1e-12) where L

	merge!(Tree)

	dict = Dict{InteractionTreeNode, Int64}() # node => (si, n)
	lsn_count = zeros(Int64, L)
	for node in PreOrderDFS(Tree.RootL)
		si, _ = node.Op
		if si == 0
			dict[node] = 1
			continue
		end

		lsn_count[si] += 1
		dict[node] = lsn_count[si]
	end
	for node in PreOrderDFS(Tree.RootR)
		si, _ = node.Op
		if si == L + 1
			dict[node] = 1
			continue
		end

		lsn_count[si] += 1
		dict[node] = lsn_count[si]
	end


	lsS = Vector{Matrix{Any}}(undef, L + 1)
	for si in 1:L-1
		lsS[si+1] = Matrix{Any}(nothing, lsn_count[si], lsn_count[si+1])
	end
	lsS[1] = ones(1, lsn_count[1])
	lsS[L+1] = ones(lsn_count[L], 1)

	# super MPS 
	lsT = Vector{Array{Any, 3}}(undef, L)
	for si in 1:L
		lsT[si] = Array{Any, 3}(nothing, lsn_count[si], length(Tree.Ops[si]), lsn_count[si])
	end

	for node in PreOrderDFS(Tree.RootL)
		si, idx = node.Op
		si == 0 && continue
		i = dict[node]
		lsT[si][i, idx, i] = 1.0

		for ch in node.children
			j = dict[ch]
			lsS[si+1][i, j] = 1.0
		end

		# Intr 
		for ch in node.Intrs
			node_R = ch.LeafR
			j = dict[node_R]
			lsS[si+1][i, j] = ch.ref[]
		end
	end

	for node in PreOrderDFS(Tree.RootR)
		si, idx = node.Op
		si > L && continue
		i = dict[node]

		lsT[si][i, idx, i] = 1.0

		for ch in node.children
			j = dict[ch]
			lsS[si][j, i] = 1.0
		end
	end

	for i in 1:L
		# reshape T 
		sz = size(lsT[i])
		T = reshape(lsT[i], sz[1], sz[2] * sz[3])

		T = _mulL(T, lsS[i])
		# reshape to left canonicalize
		T = reshape(T, size(T, 1) * sz[2], sz[3])

		# decompose T 
		T, Q = _decompose!(T; tol = tol)
		# contract Q to the right
		lsS[i+1] = _mulL(lsS[i+1], Q)

		# reshape back 
		lsT[i] = reshape(T, div(size(T, 1), sz[2]), sz[2], size(T, 2))

	end

	for i in reverse(1:L)
		# reshape T 
		sz = size(lsT[i])
		T = reshape(lsT[i], sz[1] * sz[2], sz[3])

		T = _mulR(T, lsS[i+1])
		# reshape to right canonicalize
		T = reshape(T, sz[1], sz[2] * size(T, 2))

		# decompose T 
		if i > 1
			lsS[i], T = _decompose!(T; tol = tol)
		end

		# reshape back 
		lsT[i] = reshape(T, size(T, 1), sz[2], div(size(T, 2), sz[2]))

		# check rank 
		ids = findall(1:size(lsT[i], 3)) do j
			all(isnothing, lsT[i][:, :, j])
		end
		if !isempty(ids)
			ids_keep = setdiff(1:size(lsT[i], 3), ids)
			lsT[i] = lsT[i][:, :, ids_keep]
			lsT[i+1] = lsT[i+1][ids_keep, :, :]
		end
	end

	lsH = Vector{SparseMPOTensor}(undef, L)
	for si in 1:L
		sz = size(lsT[si])
		H = SparseMPOTensor(nothing, sz[1], sz[3])
		for i in 1:sz[1], j in 1:sz[3]
			for k in 1:sz[2]
				isnothing(lsT[si][i, k, j]) && continue
				Op = deepcopy(Tree.Ops[si][k])
				Op.strength[] = lsT[si][i, k, j]
				H[i, j] += Op
			end
		end

		lsH[si] = H
	end

	return SparseMPO(lsH)
end

function _mulL(matH, matS)
	sz = (size(matS, 1), size(matH, 2))
	H_new = similar(matH, sz)
	fill!(H_new, nothing)
	for i in 1:sz[1], k in 1:sz[2]
		for j in 1:size(matH, 1)
			isnothing(matS[i, j]) && continue
			isnothing(matH[j, k]) && continue
			H_new[i, k] += matH[j, k] * matS[i, j]
		end

	end
	return H_new
end
function _mulR(matH, matS)
	sz = (size(matH, 1), size(matS, 2))
	H_new = similar(matH, sz)
	fill!(H_new, nothing)
	for i in 1:sz[1], k in 1:sz[2]
		for j in 1:size(matH, 2)
			isnothing(matS[j, k]) && continue
			isnothing(matH[i, j]) && continue
			H_new[i, k] += matH[i, j] * matS[j, k]
		end
	end
	return H_new
end
function _decompose!(S; tol::Float64 = 1e-12)
	# S = P * I_r * Q, note S will be modified
	T = mapreduce(typeof, promote_type, filter(!isnothing, S))
	P = diagm(ones(T, size(S, 1)))
	Q = diagm(ones(T, size(S, 2)))


	row_first = 1
	col_first = 1
	while row_first ≤ size(S, 1)
		# find blocks 
		ids_row = [row_first]
		ids_col = findall(!isnothing, S[row_first, :])
		if isempty(ids_col)
			row_first += 1
			continue
		end
		while true
			breakflag = true
			# find new rows 
			ids_row_add = mapreduce(union, ids_col; init = Int64[]) do j
				findall(!isnothing, S[:, j])
			end
			setdiff!(ids_row_add, ids_row)
			if !isempty(ids_row_add)
				append!(ids_row, ids_row_add)
				breakflag = false
			end
			# find new columns
			ids_col_add = mapreduce(union, ids_row) do j
				findall(!isnothing, S[j, :])
			end
			setdiff!(ids_col_add, ids_col)
			if !isempty(ids_col_add)
				append!(ids_col, ids_col_add)
				breakflag = false
			end
			breakflag && break
		end
		sort!(ids_row)
		sort!(ids_col)


		# permute columns 
		col_dest = col_first .+ (0:length(ids_col)-1)
		for (j, k) in zip(ids_col, col_dest)
			S[:, [j, k]] = S[:, [k, j]]
			Q[[j, k], :] = Q[[k, j], :]
		end
		# permute rows 
		row_dest = row_first .+ (0:length(ids_row)-1)
		for (j, k) in zip(ids_row, row_dest)
			S[[j, k], :] = S[[k, j], :]
			P[:, [j, k]] = P[:, [k, j]]
		end

		# process this block
		i = row_dest[1]
		j = col_dest[1]
		while i ≤ row_dest[end] && j ≤ col_dest[end]
			# make sure the first element is nonzero
			ids_nonzero = findall(1:size(S, 1)) do idx
				idx < i && return false
				idx > row_dest[end] && return false
				return !isnothing(S[idx, j])
			end
			if isempty(ids_nonzero)
				j += 1
				continue
			end
			# find the largest one 
			_, idx_max = findmax(abs, S[ids_nonzero, j])
			idx = ids_nonzero[idx_max]
			if idx != i
				# permute row 
				S[[i, idx], :] = S[[idx, i], :]
				P[:, [i, idx]] = P[:, [idx, i]]
			end
			# scale to 1 
			c = S[i, j]
			S[i, :] *= inv(c)
			P[:, i] *= c
			# make other elements in the column zero
			for k in i+1:row_dest[end]
				isnothing(S[k, j]) && continue
				c = S[k, j]
				S[k, :] += (-c) * S[i, :]
				# i |  1     |
				#   |    1   |
				# k |  c   1 |
				P[:, i] += c * P[:, k]

				# make sure S[k, j] = nothing 
				S[k, j] = nothing

				# change 0 to nothing 
				for idx in 1:size(S, 2)
					isnothing(S[k, idx]) && continue
					abs(S[k, idx]) < tol && (S[k, idx] = nothing)
				end
			end

			i += 1
			j += 1
		end

		# process columns 
		for i in row_dest
			j = findfirst(S[i, :]) do x
				isnothing(x) && return false
				return isapprox(x, one(T))
			end
			isnothing(j) && continue
			for k in j+1:col_dest[end]
				isnothing(S[i, k]) && continue
				c = S[i, k]
				S[i, k] = nothing
				Q[j, :] += c * Q[k, :]
			end
		end


		row_first += length(ids_row)
		col_first += length(ids_col)
	end

	# final permutations
	ids_col = findall(j -> any(!isnothing, S[:, j]), 1:size(S, 2)) |> sort!
	ids_row = findall(i -> any(!isnothing, S[i, :]), 1:size(S, 1)) |> sort!
	# permute columns 
	for (j, k) in zip(ids_col, 1:length(ids_col))
		S[:, [j, k]] = S[:, [k, j]]
		Q[[j, k], :] = Q[[k, j], :]
	end
	# permute rows 
	for (j, k) in zip(ids_row, 1:length(ids_row))
		S[[j, k], :] = S[[k, j], :]
		P[:, [j, k]] = P[:, [k, j]]
	end

	r = length(ids_row)

	Pr = [iszero(P[i, j]) ? nothing : P[i, j] for i in 1:size(P, 1), j in 1:r]
	Qr = [iszero(Q[i, j]) ? nothing : Q[i, j] for i in 1:r, j in 1:size(Q, 2)]

	return Pr, Qr

end
