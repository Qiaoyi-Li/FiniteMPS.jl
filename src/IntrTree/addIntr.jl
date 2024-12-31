"""
	 addIntr!(Root::InteractionTreeNode,
		  Op::NTuple{N,AbstractTensorMap},
		  si::NTuple{N,Int64},
		  fermionic::NTuple{N,Bool},
		  strength::Number;
		  kwargs...) 

Generic function to add an N-site interaction. `fermionic` is a tuple of bools indicating whether the corresponding operator is fermionic. If any operator is fermionic, `Z` should be provided via kwargs. For example, `addIntr!(Root, (A, B), (i, j), (false, false), s)` will add a bosonic two-site term `sAᵢBⱼ` between site `i` and `j`. 

# Kwargs
	Z::Union{Nothing, AbstractTensorMap, Vector{<:AbstractTensorMap}} = nothing
`Z = nothing` means all operators are bosonic. A single `Z::AbstractTensorMap` means all sites share the same `Z`. For systems with mixed physical spaces, please provide a vector of `Z` with length equal to the system size. 

	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing
The physical space of each site. If `pspace == nothing`, program will try to deduce it from `Z` or the first operator, which works in most cases.    

	Obs::Bool = false
This kwarg is usually added automatically by `addObs!`. `Obs == true` means this term is used for calculating observables. When adding an interaction twice, `Obs = true` will not result in a doubling of strength, and vice versa. Moreover, the information such as `name` and `si` will be stored in the leaf node additionally. 

	name::NTuple{N,Union{Symbol,String}} = (:A, :B, :C, ...)
Give a name to each operator. If `Obs == true`, the name to label the observable will be its string product. For example, setting `name = (:S, :S)` will label the observable as `"SS"`.

	value = nothing
Used for collecting observables from the tree via `convert` function, `DO NOT` set it manually. 
"""
function addIntr!(Root::InteractionTreeNode,
	Op::NTuple{N, AbstractTensorMap},
	si::NTuple{N, Int64},
	fermionic::NTuple{N, Bool},
	strength::Number;
	Obs::Bool = false,
	Z::Union{Nothing, AbstractTensorMap, Vector{<:AbstractTensorMap}} = nothing,
	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
	name::NTuple{N, Union{Symbol, String}} = _default_IntrName(N),
	value = Obs ? si => prod(string.(name)) : nothing) where N

	# convert to string
	name = string.(name)
	iszero(strength) && return nothing

	any(fermionic) && @assert !isnothing(Z)
	# even number of fermionic operators
	@assert iseven(sum(fermionic))

	# construct LocalOperators
	lsOp = map(Op, si, fermionic, name) do o, s, f, n
		LocalOperator(o, n, s, f)
	end

	S = StringOperator(lsOp..., strength) |> sort! |> reduce!

	return addIntr!(Root, S, Z; value = value, pspace = pspace)

end
addIntr!(Tree::InteractionTree, args...; kwargs...) = addIntr!(Tree.Root, args...; kwargs...)
function addIntr!(Root::InteractionTreeNode,
     Op::AbstractTensorMap,
     si::Int64,
     strength::Number;
     Obs::Bool = false,
	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
	name::Union{Symbol, String} = "A",
	value = Obs ? (si,) => string(name) : nothing)
     return addIntr!(Root, (Op,), (si,), (false,), strength; Obs = Obs, pspace = pspace, name = (name,), value = value)
end
function addIntr!(Root::InteractionTreeNode,
	S::StringOperator,
	Z::Union{Nothing, AbstractTensorMap, Vector{<:AbstractTensorMap}};
	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
	value::Union{Nothing, Pair{<:Tuple, String}} = nothing,
) 
     N = length(S)

	if isa(pspace, Vector) && isa(Z, Vector)
		@assert length(pspace) == length(Z)
	end
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

	current_node = Root
	si = 1
	idx_Op = 1
	f_flag = false
	while idx_Op ≤ N
		if si < S[idx_Op].si
			if f_flag
				Op_i = LocalOperator(_getZ(Z, si), :Z, si, false)
			else
				Op_i = IdentityOperator(_getpspace(pspace, si), si)
			end
		else
			Op_i = S[idx_Op]
			# add Z if necessary
			f_flag && _addZ!(Op_i, _getZ(Z, si))
			f_flag = xor(f_flag, isfermionic(Op_i))

			idx_Op += 1
		end

		# try to find existed node
		idx = findfirst(x -> x.Op == Op_i, current_node.children)
		if isnothing(idx)
			if si == S[end].si
				# last node
				addchild!(current_node, Op_i, value)
				current_node.children[end].Op.strength = S.strength
			else
				addchild!(current_node, Op_i)
			end
			current_node = current_node.children[end]
		else
			if si == S[end].si
				# last node
				if !isnothing(value) # obvservable 
					push!(current_node.children[idx].value, value)
				end
				_update_strength!(current_node.children[idx], S.strength) && deleteat!(current_node.children, idx)
			else
				current_node = current_node.children[idx]
			end
		end
		si += 1

	end

     return nothing
end

_getZ(::Nothing, ::Int64) = nothing
_getZ(Z::AbstractTensorMap, ::Int64) = Z
_getZ(Z::Vector{AbstractTensorMap}, i::Int64) = Z[i]
_getpspace(::Nothing, ::Int64) = nothing
_getpspace(pspace::VectorSpace, ::Int64) = pspace
_getpspace(pspace::Vector{VectorSpace}, i::Int64) = pspace[i]


function _default_IntrName(N::Int64)
	if N ≤ 26
		return [string(Char(64 + i)) for i in 1:N] |> tuple
	else
		return ["O$i" for i in 1:N] |> tuple
	end
end

function _update_strength!(node::InteractionTreeNode, strength::Number)
	if isnan(node.Op.strength)
		node.Op.strength = strength
	else
		node.Op.strength += strength
	end
	# delete this channel if strength == 0
	if node.Op.strength == 0
		# delete
		isempty(node.children) && return true # delete or not
		# propagate
		node.Op.strength = NaN
	end
	return false
end
function _update_strength!(node::InteractionTreeNode{T}, strength::Number) where T <: Dict{<:Tuple, String}
	# used for calculating observables
	if isnan(node.Op.strength)
		node.Op.strength = strength
	else
		@assert node.Op.strength == strength
	end
	return false
end

_addZ!(O::LocalOperator, ::Nothing) = O
function _addZ!(OR::LocalOperator{1, 1}, Z::AbstractTensorMap)
	OR.A = Z * OR.A
	if OR.name[1] == 'Z'
		OR.name = OR.name[2:end]
	else
		OR.name = "Z" * OR.name
	end
	return OR
end
function _addZ!(OR::LocalOperator{1, R₂}, Z::AbstractTensorMap) where R₂
	OR.A = Z * OR.A
	if OR.name[1] == 'Z'
		OR.name = OR.name[2:end]
	else
		OR.name = "Z" * OR.name
	end
	return OR
end
function _addZ!(OR::LocalOperator{R₁, 1}, Z::AbstractTensorMap) where R₁
	# note ZA = - AZ
	OR.A = -OR.A * Z
	if OR.name[1] == 'Z'
		OR.name = OR.name[2:end]
	else
		OR.name = "Z" * OR.name
	end
	return OR
end

function _addZ!(OR::LocalOperator{2, 2}, Z::AbstractTensorMap)
	@tensor OR.A[a e; c d] := Z[e b] * OR.A[a b c d]
	if OR.name[1] == 'Z'
		OR.name = OR.name[2:end]
	else
		OR.name = "Z" * OR.name
	end
	return OR
end

# swap two operators to deal with horizontal bond
_swap(A::LocalOperator{1, 1}, B::LocalOperator{1, 1}) = B, A
_swap(A::LocalOperator{1, 1}, B::LocalOperator{1, 2}) = B, A
_swap(A::LocalOperator{1, 2}, B::LocalOperator{1, 1}) = B, A
_swap(A::LocalOperator{2, 1}, B::LocalOperator{1, 1}) = B, A
_swap(A::LocalOperator{1, 1}, B::LocalOperator{2, 1}) = B, A
function _swap(A::LocalOperator{1, 2}, B::LocalOperator{2, 1})
	return _swapOp(B), _swapOp(A)
end
function _swap(A::LocalOperator{1, 2}, B::LocalOperator{2, 2})
	#  |      |          |      |
	#  A--  --B--va -->  B--  --A--va
	#  |      |          |      |

	@tensor AB[d e; a b f] := A.A[a b c] * B.A[c d e f]
	# QR 
	TA, TB = leftorth(AB)

	return LocalOperator(permute(TA, (1,), (2, 3)), B.name, B.si, B.fermionic, B.strength), LocalOperator(permute(TB, (1, 2), (3, 4)), A.name, A.si, A.fermionic, A.strength)
end
function _swap(A::LocalOperator{2, 2}, B::LocalOperator{2, 1})
	#     |     |         |     |
	# va--A-- --B --> va--B-- --A 
	#     |     |         |     |

	@tensor AB[a e f; b c] := A.A[a b c d] * B.A[d e f]
	# QR
	TA, TB = rightorth(AB)

	return LocalOperator(permute(TA, (1, 2), (3, 4)), B.name, B.si, B.fermionic, B.strength), LocalOperator(permute(TB, (1, 2), (3,)), A.name, A.si, A.fermionic, A.strength)
end
function _swap(A::LocalOperator{2, 2}, B::LocalOperator{2, 2})
	#     |     |             |     |
	# va--A-- --B--vb --> va--B-- --A--vb 
	#     |     |             |     |

	@tensor AB[a e f; b c g] := A.A[a b c d] * B.A[d e f g]
	# QR
	TA, TB = leftorth(AB)

	return LocalOperator(permute(TA, (1, 2), (3, 4)), B.name, B.si, B.fermionic, B.strength), LocalOperator(permute(TB, (1, 2), (3, 4)), A.name, A.si, A.fermionic, A.strength)
end
function _swap(A::LocalOperator{2, 1}, B::LocalOperator{1, 2})
	#     |   |             |     |
	# va--A   B--vb --> va--B-- --A--vb 
	#     |   |             |     |

	@tensor AB[a e f; b c g] := A.A[a b c] * B.A[e f g]
	# QR
	TA, TB = leftorth(AB)

	return LocalOperator(permute(TA, (1, 2), (3, 4)), B.name, B.si, B.fermionic, B.strength), LocalOperator(permute(TB, (1, 2), (3, 4)), A.name, A.si, A.fermionic, A.strength)
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
