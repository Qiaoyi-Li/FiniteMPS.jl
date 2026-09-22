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

Add an interaction characterized by `N` local operators at `si = (i, j, ...)` sites. `fermionic` indicates whether each operator is fermionic or not. Chains may be closed or open at one end; right-open chains are normalized to left-open form before insertion.

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

	S = StringOperator(collect(_local_operator_chain(Op, si, fermionic, name)), strength) |> sort! |> reduce!

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


	Ops_idx = map(_chain_iterator(S, Z, pspace, Val(L))) do Op
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


	return _insert_paired_channel!(Tree, Ops_idx, ref)
end

function _default_IntrName(N::Int64)
	if N ≤ 26
		return [string(Char(64 + i)) for i in 1:N] |> Tuple
	else
		return ["O$i" for i in 1:N] |> Tuple
	end
end


function _local_operator_chain(ops, sites, fermionic, names)
    isempty(ops) && throw(ArgumentError("an operator chain cannot be empty"))
    length(ops) == length(sites) == length(fermionic) == length(names) ||
        throw(ArgumentError("operators, sites, fermionic flags and names must have equal lengths"))
    auxiliary = unitspace(codomain(first(ops))[1])
    return map(ops, sites, fermionic, names) do op, site, flag, name
        localop = LocalOperator(op, string(name), site, flag; aspace = (auxiliary, auxiliary))
        auxiliary = getRightSpace(localop)
        return localop
    end
end

function _chain_iterator(S::StringOperator, Z, pspace, ::Val{L}) where L
    pspace isa AbstractVector && length(pspace) != L && throw(ArgumentError("pspace must have length $L"))
    Z isa AbstractVector && length(Z) != L && throw(ArgumentError("Z must have length $L"))
    all(op -> 1 <= op.si <= L, S.Ops) || throw(ArgumentError("operator sites must lie in 1:$L"))
    _normalize_right_boundary!(S)
    if isnothing(pspace)
        pspace = Z isa AbstractVector ? map(z -> codomain(z)[1], Z) :
            Z isa AbstractTensorMap ? codomain(Z)[1] : getPhysSpace(first(S))
    end
    return ArbitraryInteractionIterator{L}(S.Ops, Z, pspace)
end
