"""
	AutomataMPO(Tree::InteractionTree;
		compress::Int64 = 1,
		tol::Float64 = 1e-12
	) -> H::SparseMPO

Generate the Hamiltonian MPO `H::SparseMPO` according to an `InteractionTree`. 

# Kwargs 
	compress::Int64 = 1
The compress level. `0`: without compression. `1(default)`: merge the same rows/columns. `2`: compress via lu decomposition. `3`: compress via svd. Note higher `compress` results in lower MPO bond dimension, however may lead to more operators in the sparse matrix at each site.

	tol::Float64 = 1e-12
Tolerance of the low-rank decompositions.
"""
function AutomataMPO(Tree::InteractionTree{L}; compress::Int64 = 1, tol::Float64 = 1e-12) where L

	lsT, lsS = _superMPS(Tree)

	lsT = _compress0!(lsT, lsS)

	if compress == 1
		lsT = _compress1!(lsT; tol = tol)
	elseif compress == 2
		lsT = _compress2!(lsT; tol = tol)
	elseif compress == 3
		lsT = _compress3!(lsT; tol = tol)
	else
		!iszero(compress) && @error "invalid compress level $(compress)!"
	end

	# write operators to MPO
	lsH = Vector{SparseMPOTensor}(undef, L)
	for si in 1:L
		sz = size(lsT[si])
		H = SparseMPOTensor(nothing, sz[1], sz[3])
		for i in 1:sz[1], j in 1:sz[3]
			for k in 1:sz[2]
				iszero(lsT[si][i, k, j]) && continue
				Op = deepcopy(Tree.Ops[si][k])
				s = lsT[si][i, k, j]
				# try to use Float64 
				if abs(imag(s)) < tol 
					s = real(s)
				end
				Op.strength[] = s
				H[i, j] += Op
			end
		end

		lsH[si] = H
	end

	return SparseMPO(lsH)

end

function _superMPS(Tree::InteractionTree{L}) where L

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

	FF = Float64
	for d in values(Tree.Refs)
		for (_, v) in d 
			FF = promote_type(FF, typeof(v[]))
			FF <: Complex && break
		end
		FF <: Complex && break
	end

	lsS = Vector{SparseMatrixCSC{FF}}(undef, L + 1)
	for si in 1:L-1
		lsS[si+1] = zeros(FF, lsn_count[si], lsn_count[si+1])
	end
	lsS[1] = ones(FF, 1, lsn_count[1])
	lsS[L+1] = ones(FF, lsn_count[L], 1)

	# super MPS 
	lsT = Vector{Array}(undef, L)
	for si in 1:L
		lsT[si] = zeros(lsn_count[si], length(Tree.Ops[si]), lsn_count[si])
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

	return lsT, lsS
end

function _compress0!(lsT::Vector{Array}, lsS::Vector{<:SparseMatrixCSC})
	# directly contract S tensor to T tensor
	L = length(lsT)
	@tensor lsT[1][a c d] := lsS[1][a b] * lsT[1][b c d]
	@tensor lsT[end][a b d] := lsT[end][a b c] * lsS[end][c d]
	for i in 1:L-1
		sz = size(lsS[i+1])
		if sz[1] ≤ sz[2]
			@tensor lsT[i+1][a c d] := lsS[i+1][a b] * lsT[i+1][b c d]
		else
			@tensor lsT[i][a b d] := lsT[i][a b c] * lsS[i+1][c d]
		end
	end

	return lsT
end

function _compress1!(lsT::Vector{Array}; tol::Float64 = 1e-12)
	# allow scalar multiplication, merge same rows/columns
	L = length(lsT)

	# right-to-left sweep
	for si in reverse(2:L)
		T = reshape(lsT[si], size(lsT[si], 1), size(lsT[si], 2) * size(lsT[si], 3)) |> sparse

		# find same rows 
		P = zeros(eltype(T), size(T, 1), 0) |> sparse
		ids_zero = Int64[]
		for i in 1:size(T, 1)
			in(i, ids_zero) && continue
			norm_i = norm(T[i, :])
			if norm_i < tol
				push!(ids_zero, i)
				continue
			end
			# add a column to P
			P = hcat(P, zeros(size(P, 1), 1))
			P[i, end] = 1.0

			ids = findall(1:size(T, 1)) do k
				k ≤ i && return false
				norm_k = norm(T[k, :])
				norm_k < tol && return false
				# w == cv iff |⟨w,v⟩| = |w||v|
				return abs(abs(dot(T[i, :], T[k, :]) / norm_i / norm_k) - 1.0) < tol
			end

			for k in ids
				# w = cv => w = v ⟨v, w⟩ / |v|^2
				P[k, end] = dot(T[i, :], T[k, :]) / norm_i^2
				push!(ids_zero, k)
			end
		end

		# remove zero rows
		if !isempty(ids_zero)
			ids_keep = setdiff(1:size(T, 1), ids_zero)
			T = T[ids_keep, :]
		end

		# update
		lsT[si] = reshape(T, :, size(lsT[si], 2), size(lsT[si], 3))
		@tensor lsT[si-1][a b d] := lsT[si-1][a b c] * P[c d]
	end

	# left-to-right sweep 
	for si in 1:L-1
		T = reshape(lsT[si], size(lsT[si], 1) * size(lsT[si], 2), size(lsT[si], 3)) |> sparse

		# find same columns (up tp a scalar)
		Q = zeros(eltype(T), 0, size(T, 2)) |> sparse
		ids_zero = Int64[]
		for j in 1:size(T, 2)
			in(j, ids_zero) && continue

			norm_j = norm(T[:, j])
			if norm_j < tol
				push!(ids_zero, j)
				continue
			end
			# add a row to Q 
			Q = vcat(Q, zeros(1, size(Q, 2)))
			Q[end, j] = 1.0
			ids = findall(1:size(T, 2)) do k
				k ≤ j && return false
				norm_k = norm(T[:, k])
				norm_k < tol && return false
				# w == cv iff |⟨w,v⟩| = |w||v|
				return abs(abs(dot(T[:, j], T[:, k]) / norm_j / norm_k) - 1.0) < tol
			end

			for k in ids
				# w = cv => w = v ⟨v, w⟩ / |v|^2
				Q[end, k] = dot(T[:, j], T[:, k]) / norm_j^2
				push!(ids_zero, k)
			end
		end

		# remove zero columns 
		if !isempty(ids_zero)
			ids_keep = setdiff(1:size(T, 2), ids_zero)
			T = T[:, ids_keep]
		end

		# update 
		lsT[si] = reshape(T, size(lsT[si], 1), size(lsT[si], 2), :)

		@tensor lsT[si+1][a c d] := Q[a b] * lsT[si+1][b c d]
	end

	return lsT
end

function _compress2!(lsT::Vector{Array}; tol::Float64 = 1e-12)
	# low-rank decomposition via sparse lu decomposition
	L = length(lsT)

	# right-to-left sweep
	for si in reverse(2:L)
		T = reshape(lsT[si], size(lsT[si], 1), :) |> sparse

		l, u = _lu_wrap(T; tol = tol)

		lsT[si] = reshape(u, :, size(lsT[si], 2), size(lsT[si], 3))
		@tensor lsT[si-1][a b d] := lsT[si-1][a b c] * l[c d]
	end

	# left-to-right sweep
	for si in 1:L-1
		T = reshape(lsT[si], size(lsT[si], 1) * size(lsT[si], 2), :) |> sparse

		# T' = lu => T = u'l'
		l, u = _lu_wrap(T'; tol = tol)
		l, u = u', l'

		lsT[si] = reshape(l, size(lsT[si], 1), size(lsT[si], 2), size(l, 2))
		@tensor lsT[si+1][a c d] := u[a b] * lsT[si+1][b c d]
	end

	return lsT
end

function _compress3!(lsT::Vector{Array}; tol::Float64 = 1e-12)
	# compress via svd
	L = length(lsT)
	
	# right-to-left sweep 
	for si in reverse(2:L)
		T = reshape(lsT[si], size(lsT[si], 1), :) 
		u, s, vd = _svd_wrap!(T; tol = eps(Float64))
		l = u*s

		lsT[si] = reshape(vd, :, size(lsT[si], 2), size(lsT[si], 3))
		@tensor lsT[si-1][a b d] := lsT[si-1][a b c] * l[c d]
	end

	# left-to-right sweep with compression
	for si in 1:L-1
		T = reshape(lsT[si], size(lsT[si], 1) * size(lsT[si], 2), :)

		u, s, vd = _svd_wrap!(T; tol = tol)
		r = s*vd

		lsT[si] = reshape(u, size(lsT[si], 1), size(lsT[si], 2), size(u, 2))
		@tensor lsT[si+1][a c d] := r[a b] * lsT[si+1][b c d]
	end

	return lsT
end

function _lu_wrap(A::AbstractMatrix; tol::Float64 = 1e-12)::NTuple{2, <:SparseMatrixCSC}

	l, u = try
		rslt = lu(A)
		# L*U = (diagm(Rs) * A)[p, q]
		p_inv = invperm(rslt.p)
		q_inv = invperm(rslt.q)
		Rs_inv = 1 ./ rslt.Rs
		l = Rs_inv .* rslt.L[p_inv, :]
		u = rslt.U[:, q_inv]
		@assert norm(l * u - A) < tol * norm(A)
		l, u
	catch e
		# not full rank, use dense lu with allowsingular = true
		# l * u = A[p, :]
		rslt = lu(Matrix(A); allowsingular = true)
		p_inv = invperm(rslt.p)
		l = rslt.L[p_inv, :]
		u = rslt.U
		@assert norm(l * u - A) < tol * norm(A)
		l, u
	end

	# skip noise
	norm_col = map(1:size(l, 2)) do j
		norm(l[:, j])
	end
	norm_row = map(1:size(u, 1)) do i
		norm(u[i, :])
	end
	ids_keep = filter(1:length(norm_col)) do i
		norm_col[i] > tol && norm_row[i] > tol
	end

	return l[:, ids_keep], u[ids_keep, :]
end


function _svd_wrap!(A::AbstractMatrix; tol::Float64 = 1e-12)
	sz = size(A)
	U = zeros(eltype(A), sz[1], minimum(sz))
	S = zeros(Float64, minimum(sz))
	Vd = zeros(eltype(A), minimum(sz), sz[2])

	# block diagonal, A_f = A_i[p, q]
	p = collect(1:sz[1])
	q = collect(1:sz[2])

	row_first = 1
	col_first = 1
	idx_S = 0
	while row_first ≤ size(A, 1)
		ids_row = [row_first]
		ids_col = findall(!iszero, A[row_first, :])
		if isempty(ids_col)
			row_first += 1
			continue
		end
		while true
			breakflag = true
			# find new rows 
			ids_row_add = mapreduce(union, ids_col; init = Int64[]) do j
				findall(!iszero, A[:, j])
			end
			setdiff!(ids_row_add, ids_row)
			if !isempty(ids_row_add)
				append!(ids_row, ids_row_add)
				breakflag = false
			end
			# find new columns
			ids_col_add = mapreduce(union, ids_row) do j
				findall(!iszero, A[j, :])
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
			A[:, [j, k]] = A[:, [k, j]]
			q[[j, k]] = q[[k, j]]
		end
		# permute rows 
		row_dest = row_first .+ (0:length(ids_row)-1)
		for (j, k) in zip(ids_row, row_dest)
			A[[j, k], :] = A[[k, j], :]
			p[[j, k]] = p[[k, j]]
		end

		# svd 
		u, s, v = svd(A[row_dest, col_dest])
		S_dest = idx_S .+ (1:length(s))
		U[row_dest, S_dest] .= u
		S[S_dest] .= s 
		Vd[S_dest, col_dest] .= v'

		# copyto!(U, row_dest, S_dest, 'N',
		# 	u, 1:size(u, 1), 1:size(u, 2))
		# copyto!(S, S_dest, s, 1:length(s))
		# copyto!(Vd, S_dest, col_dest, 'C',
		# 	v, 1:size(v, 2), 1:size(v, 1)
		# )

		row_first += length(ids_row)
		col_first += length(ids_col)
		idx_S += sz[1] < sz[2] ? length(ids_row) : length(ids_col)
	end

	# truncate 
	norm_S = norm(S)
	ids_keep = findall(x -> x > norm_S * tol, S)

	# permute back 
	U = U[invperm(p), ids_keep]
	Vd = Vd[ids_keep, invperm(q)]
	S = S[ids_keep]

	return sparse(U), sparse(diagm(S)), sparse(Vd)
end