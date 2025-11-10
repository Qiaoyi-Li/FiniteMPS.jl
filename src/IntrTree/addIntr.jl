"""
	addIntr!(Tree::InteractionTree{L},
		Op::NTuple{N, AbstractTensorMap},
		si::NTuple{N, Int64},
		fermionic::NTuple{N, Bool},
		strength::Number;
		Z = nothing,
		pspace = nothing,
		name = _default_IntrName(N),
		IntrName = prod(string.(name)),
	) -> nothing

Add an interaction characterized by `N` local operators at `si = (i, j, ...)` sites. `fermionic` indicates whether each operator is fermionic or not. 

# Kwargs 
	Z::Union{Nothing, AbstractTensorMap, AbstractVector}
Provide the parity operator to deal with the fermionic anti-commutation relations.If `Z == nothing`, assume all operators are bosonic. Otherwise, a uniform (single operator) `Z::AbstractTensorMap` or site-dependent (length `L` vector) `Z::AbstractVector` should be given.

	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}}
Provide the local Hilbert space (`VectorSpace` in `TensorKit.jl`). This is not required in generating Hamiltonian, so the default value is set as `nothing`. But some processes like generating an identity MPO require this information. In such cases, a uniform or site-dependent (length `L` vector) `pspace` should be given.

	name::NTuple{N, Union{Symbol, String}}
Give a name of each operator.

	IntrName::Union{Symbol, String}
Give a name of the interaction, which is used as the key of `Tree.Refs::Dict` that stores interaction strengths. The default value is the product of each operator name.
"""
function addIntr!(Tree::InteractionTree{L},
	Op::NTuple{N, AbstractTensorMap},
	si::NTuple{N, Int64},
	fermionic::NTuple{N, Bool},
	strength::Number;
	Z::Union{Nothing, AbstractTensorMap, AbstractVector} = nothing,
	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
	name::NTuple{N, Union{Symbol, String}} = _default_IntrName(N),
	IntrName::Union{Symbol, String} = prod(string.(name)),
) where {L, N}

	# convert to string
	name = string.(name)
	IntrName = string(IntrName)
	iszero(strength) && return nothing

	any(fermionic) && @assert !isnothing(Z)

	# update Refs
	if !haskey(Tree.Refs, IntrName)
		Tree.Refs[IntrName] = Dict{NTuple{N, Int64}, Ref{Number}}()
	end

	# construct LocalOperators
	aspace = trivial(codomain(Op[1])[1])
	lsOp = map(Op, si, fermionic, name) do o, s, f, n
		Oi = LocalOperator(o, n, s, f; aspace = (aspace, aspace)) 
		aspace = getRightSpace(Oi) # update the horizontal space for the next operator
		return Oi
	end
	S = StringOperator(lsOp..., strength) |> sort! |> reduce!

	# existed key
	if haskey(Tree.Refs[IntrName], si)
		Tree.Refs[IntrName][si][] += S.strength
		return nothing
	else
		Tree.Refs[IntrName][si] = Ref{Number}(S.strength)
		return addIntr!(Tree, S, Z, Tree.Refs[IntrName][si]; pspace = pspace)
	end

end

"""
	addIntr!(Tree::InteractionTree{L},
		Op::AbstractTensorMap,
		si::Int64,
		strength::Number;
		pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
		name::Union{Symbol, String} = "A",
		IntrName::Union{Symbol, String} = string.(name),
	) 

The special case for on-site interactions. Compared with the standard usage, converting to tuples is not required for convenience.
"""
function addIntr!(Tree::InteractionTree{L},
	Op::AbstractTensorMap,
	si::Int64,
	strength::Number;
	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
	name::Union{Symbol, String} = "A",
	IntrName::Union{Symbol, String} = string.(name),
) where L
	return addIntr!(Tree, (Op,), (si,), (false,), strength; pspace = pspace, name = (name,), IntrName = IntrName)
end

function addIntr!(Tree::InteractionTree{L},
	S::StringOperator,
	Z::Union{Nothing, AbstractTensorMap, AbstractVector},
	ref::Ref;
	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
) where L

	isa(pspace, Vector) && @assert length(pspace) == L
	isa(Z, Vector) && @assert length(Z) == L

	# try to deduce the pspace
	if isnothing(pspace)
		if isa(Z, Vector)
			pspace = map(Z) do Zi
				codomain(Zi)[1]
			end
		elseif isa(Z, AbstractTensorMap)
			pspace = codomain(Z)[1]
		else
			pspace = getPhysSpace(S[1])
		end
	end

	Ops_idx = map(ArbitraryInteractionIterator{L}(S.Ops, Z, pspace)) do Op
		# find existed Op 
		si = Op.si
		idx = findfirst(x -> x == Op, Tree.Ops[si])
		if isnothing(idx)
			push!(Tree.Ops[si], Op)
			return length(Tree.Ops[si])
		else
			return idx
		end
	end


	nodeL = Tree.RootL
	nodeR = Tree.RootR
	flagL = flagR = false
	for i in 1:L
		# left to right 
		if !flagL
			si = i
			idx = findfirst(nodeL.children) do child
				child.Op == (si, Ops_idx[si])
			end

			if !isnothing(idx)
				nodeL = nodeL.children[idx]
			else
				# merge
				idx = findfirst(nodeL.Intrs) do ch
					length(ch.Ops) < 1 && return false
					ch.Ops[1] == Ops_idx[si]
				end
				if isnothing(idx)
					flagL = true
				else
					node_new = InteractionTreeNode((si, Ops_idx[si]), nodeL)
					ch = nodeL.Intrs[idx]
					push!(node_new.Intrs, ch)
					deleteat!(nodeL.Intrs, idx) # remove ch from nodeL
					# update ch 
					deleteat!(ch.Ops, 1)
					ch.LeafL = node_new

					push!(nodeL.children, node_new)
					nodeL = node_new
				end
			end
		end

		nodeL.Op[1] + 1 ≥ nodeR.Op[1] && break

		# right to left
		if !flagR
			si = L - i + 1
			idx = findfirst(nodeR.children) do child
				child.Op == (si, Ops_idx[si])
			end

			if !isnothing(idx)
				nodeR = nodeR.children[idx]
			else
				# merge
				idx = findfirst(nodeR.Intrs) do ch
					length(ch.Ops) < 1 && return false
					ch.Ops[end] == Ops_idx[si]
				end
				if isnothing(idx)
					flagR = true
				else
					node_new = InteractionTreeNode((si, Ops_idx[si]), nodeR)
					ch = nodeR.Intrs[idx]
					push!(node_new.Intrs, ch)
					deleteat!(nodeR.Intrs, idx) # remove ch from nodeR
					# update ch
					deleteat!(ch.Ops, length(ch.Ops))
					ch.LeafR = node_new

					push!(nodeR.children, node_new)
					nodeR = node_new
				end
			end
		end

		nodeL.Op[1] + 1 ≥ nodeR.Op[1] && break
	end

	ch = InteractionChannel(Ops_idx[nodeL.Op[1]+1:nodeR.Op[1]-1], ref, nodeL, nodeR)
	push!(nodeL.Intrs, ch)
	push!(nodeR.Intrs, ch)

	return nothing
end

function _default_IntrName(N::Int64)
	if N ≤ 26
		return [string(Char(64 + i)) for i in 1:N] |> Tuple
	else
		return ["O$i" for i in 1:N] |> Tuple
	end
end


# fuse the possible multiple bonds between two operators
function _reduceOp(A::LocalOperator{1, R}, B::LocalOperator{R, 1}) where R
	# match tags 
	tagA = A.tag[2][2:end]
	tagB = B.tag[1][1:end-1]
	permB = map(tagA) do t
		findfirst(x -> x == t, tagB)
	end
	pA = ((1, 2), Tuple(3:R+1))
	pB = (permB, (R, R + 1))
	pC = ((1, 2), (3, 4))
	AB = tensorcontract(pC, A.A, pA, :N, B.A, pB, :N)
	# QR 
	TA, TB = leftorth(AB)
	if !isnan(A.strength) && !isnan(B.strength)
		sA = 1
		sB = A.strength * B.strength
	else
		sA = sB = NaN
	end
	OA = LocalOperator(permute(TA, (1,), (2, 3)), A.name, A.si, sA)
	OB = LocalOperator(permute(TB, (1, 2), (3,)), B.name, B.si, sB)
	return OA, OB
end
_reduceOp(A::LocalOperator{1, 1}, B::LocalOperator{1, 1}) = A, B
_reduceOp(A::LocalOperator{1, 2}, B::LocalOperator{2, 1}) = A, B
# contract the possible outer bonds between 3 operators
# C AD B case from 4-site terms
#   |     |     |
# --A-- --B-- --C--   
#   |     |     |
function _reduceOp(A::LocalOperator{2, 2}, B::LocalOperator{2, 2}, C::LocalOperator{2, 2})
	@tensor ABC[b c; e f h i] := A.A[a b c d] * B.A[d e f g] * C.A[g h i a]
	# QR
	TA, TBC = leftorth(ABC)
	TB, TC = leftorth(permute(TBC, (1, 2, 3), (4, 5)))
	# strength
	if isnan(A.strength) || isnan(B.strength) || isnan(C.strength)
		sA = sB = sC = NaN
	else
		sA = sB = 1
		sC = A.strength * B.strength * C.strength
	end
	OA = LocalOperator(permute(TA, (1,), (2, 3)), A.name, A.si, sA)
	OB = LocalOperator(permute(TB, (1, 2), (3, 4)), B.name, B.si, sB)
	OC = LocalOperator(permute(TC, (1, 2), (3,)), C.name, C.si, sC)
	return OA, OB, OC
end
_reduceOp(A::LocalOperator{1, 2}, B::LocalOperator{2, 2}, C::LocalOperator{2, 1}) = A, B, C
_reduceOp(A::LocalOperator{1, 1}, B::LocalOperator{1, 1}, C::LocalOperator{1, 1}) = A, B, C
