function addIntr!(Tree::InteractionTree{L},
	Op::NTuple{N, AbstractTensorMap},
	si::NTuple{N, Int64},
	fermionic::NTuple{N, Bool},
	strength::Number;
	Z::Union{Nothing, AbstractTensorMap, Vector{<:AbstractTensorMap}} = nothing,
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
	Z::Union{Nothing, AbstractTensorMap, Vector{<:AbstractTensorMap}},
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

# """
# 	 addIntr!(Root::InteractionTreeNode,
# 		  Op::NTuple{N,AbstractTensorMap},
# 		  si::NTuple{N,Int64},
# 		  fermionic::NTuple{N,Bool},
# 		  strength::Number;
# 		  kwargs...) 

# Generic function to add an N-site interaction. `fermionic` is a tuple of bools indicating whether the corresponding operator is fermionic. If any operator is fermionic, `Z` should be provided via kwargs. For example, `addIntr!(Root, (A, B), (i, j), (false, false), s)` will add a bosonic two-site term `sAᵢBⱼ` between site `i` and `j`. 

# # Kwargs
# 	Z::Union{Nothing, AbstractTensorMap, Vector{<:AbstractTensorMap}} = nothing
# `Z = nothing` means all operators are bosonic. A single `Z::AbstractTensorMap` means all sites share the same `Z`. For systems with mixed physical spaces, please provide a vector of `Z` with length equal to the system size. 

# 	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing
# The physical space of each site. If `pspace == nothing`, program will try to deduce it from `Z` or the first operator, which works in most cases.    

# 	Obs::Bool = false
# This kwarg is usually added automatically by `addObs!`. `Obs == true` means this term is used for calculating observables. When adding an interaction twice, `Obs = true` will not result in a doubling of strength, and vice versa. Moreover, the information such as `name` and `si` will be stored in the leaf node additionally. 

# 	name::NTuple{N,Union{Symbol,String}} = (:A, :B, :C, ...)
# Give a name to each operator. If `Obs == true`, the name to label the observable will be its string product. For example, setting `name = (:S, :S)` will label the observable as `"SS"`.

# 	value = nothing
# Used for collecting observables from the tree via `convert` function, `DO NOT` set it manually. 
# """
# function addIntr!(Root::InteractionTreeNode,
# 	Op::NTuple{N, AbstractTensorMap},
# 	si::NTuple{N, Int64},
# 	fermionic::NTuple{N, Bool},
# 	strength::Number;
# 	Obs::Bool = false,
# 	Z::Union{Nothing, AbstractTensorMap, Vector{<:AbstractTensorMap}} = nothing,
# 	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
# 	name::NTuple{N, Union{Symbol, String}} = _default_IntrName(N),
# 	value = Obs ? si => prod(string.(name)) : nothing) where N

# 	# convert to string
# 	name = string.(name)
# 	iszero(strength) && return nothing

# 	any(fermionic) && @assert !isnothing(Z)
# 	# even number of fermionic operators
# 	@assert iseven(sum(fermionic))

# 	# construct LocalOperators
# 	lsOp = map(Op, si, fermionic, name) do o, s, f, n
# 		LocalOperator(o, n, s, f)
# 	end

# 	S = StringOperator(lsOp..., strength) |> sort! |> reduce!

# 	return addIntr!(Root, S, Z; value = value, pspace = pspace)

# end
# addIntr!(Tree::InteractionTree, args...; kwargs...) = addIntr!(Tree.Root, args...; kwargs...)
# function addIntr!(Tree::InteractionTree,
#      Op::AbstractTensorMap,
#      si::Int64,
#      strength::Number;
#      Obs::Bool = false,
# 	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
# 	name::Union{Symbol, String} = "A",
# 	value = Obs ? (si,) => string(name) : nothing)
#      return addIntr!(Tree, (Op,), (si,), (false,), strength; Obs = Obs, pspace = pspace, name = (name,), value = value)
# end

# function addIntr!(Root::InteractionTreeNode,
# 	S::StringOperator,
# 	Z::Union{Nothing, AbstractTensorMap, Vector{<:AbstractTensorMap}};
# 	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
# 	value::Union{Nothing, Pair{<:Tuple, String}} = nothing,
# ) 
#      N = length(S)

# 	if isa(pspace, Vector) && isa(Z, Vector)
# 		@assert length(pspace) == length(Z)
# 	end
# 	# try to deduce the pspace
# 	if isnothing(pspace)
# 		if isa(Z, Vector)
# 			pspace = map(Z) do Zi
# 				codomain(Zi)[1]
# 			end
# 		elseif isa(Z, AbstractTensorMap)
# 			pspace = codomain(Z)[1]
# 		else
# 			pspace = getPhysSpace(S[1])
# 		end
# 	end

# 	current_node = Root
# 	si = 1
# 	idx_Op = 1
# 	f_flag = false
# 	while idx_Op ≤ N
# 		if si < S[idx_Op].si
# 			if f_flag
# 				Op_i = LocalOperator(_getZ(Z, si), :Z, si, false)
# 			else
# 				Op_i = IdentityOperator(_getpspace(pspace, si), si)
# 			end
# 		else
# 			Op_i = S[idx_Op]
# 			# add Z if necessary
# 			f_flag && _addZ!(Op_i, _getZ(Z, si))
# 			f_flag = xor(f_flag, isfermionic(Op_i))

# 			idx_Op += 1
# 		end

# 		# try to find existed node
# 		idx = findfirst(x -> x.Op == Op_i, current_node.children)
# 		if isnothing(idx)
# 			if si == S[end].si
# 				# last node
# 				addchild!(current_node, Op_i, value)
# 				current_node.children[end].Op.strength = S.strength
# 			else
# 				addchild!(current_node, Op_i)
# 			end
# 			current_node = current_node.children[end]
# 		else
# 			if si == S[end].si
# 				# last node
# 				if !isnothing(value) # obvservable 
# 					push!(current_node.children[idx].value, value)
# 				end
# 				_update_strength!(current_node.children[idx], S.strength) && deleteat!(current_node.children, idx)
# 			else
# 				current_node = current_node.children[idx]
# 			end
# 		end
# 		si += 1

# 	end

#      return nothing
# end


function _default_IntrName(N::Int64)
	if N ≤ 26
		return [string(Char(64 + i)) for i in 1:N] |> tuple
	else
		return ["O$i" for i in 1:N] |> tuple
	end
end

# function _update_strength!(node::InteractionTreeNode, strength::Number)
# 	if isnan(node.Op.strength)
# 		node.Op.strength = strength
# 	else
# 		node.Op.strength += strength
# 	end
# 	# delete this channel if strength == 0
# 	if node.Op.strength == 0
# 		# delete
# 		isempty(node.children) && return true # delete or not
# 		# propagate
# 		node.Op.strength = NaN
# 	end
# 	return false
# end
# function _update_strength!(node::InteractionTreeNode{T}, strength::Number) where T <: Dict{<:Tuple, String}
# 	# used for calculating observables
# 	if isnan(node.Op.strength)
# 		node.Op.strength = strength
# 	else
# 		@assert node.Op.strength == strength
# 	end
# 	return false
# end

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
