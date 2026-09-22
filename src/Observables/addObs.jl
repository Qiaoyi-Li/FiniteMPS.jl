"""
	addObs!(Tree::ObservableTree{L},
		Op::NTuple{N, AbstractTensorMap},
		si::NTuple{N, Int64},
		fermionic::NTuple{N, Bool};
		Z = nothing,
		pspace = nothing,
		name = _default_IntrName(N),
		IntrName = prod(string.(name)),
	) -> nothing 

Add an `N`-site observable to `Tree`, where the observable is characterized by `N`-tuples `Op`, `si` and `fermionic`. Registration and boundary normalization follow `addIntr!`, except for the logic to deal with the same observable added twice.

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
function addObs!(Tree::ObservableTree{L},
	Op::NTuple{N, AbstractTensorMap},
	si::NTuple{N, Int64},
	fermionic::NTuple{N, Bool};
	Z::Union{Nothing, AbstractTensorMap, AbstractVector} = nothing,
	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
	name::NTuple{N, Union{Symbol, String}} = _default_IntrName(N),
	IntrName::Union{Symbol, String} = prod(string.(name)),
) where {L, N}

	# convert to string
	name = string.(name)
	IntrName = string(IntrName)

	any(fermionic) && @assert !isnothing(Z)

	# update Refs
	if !haskey(Tree.Refs, IntrName)
		Tree.Refs[IntrName] = Dict{NTuple{N, Int64}, Ref{Number}}()
	end
     haskey(Tree.Refs[IntrName], si) && return nothing

	S = StringOperator(collect(_local_operator_chain(Op, si, fermionic, name))) |> sort! |> reduce!

	Tree.Refs[IntrName][si] = Ref{Number}()
	return addObs!(Tree, S, Z, Tree.Refs[IntrName][si]; pspace = pspace)
end

function addObs!(Tree::ObservableTree{L},
	Op::AbstractTensorMap,
	si::Int64;
	pspace::Union{Nothing, VectorSpace, Vector{<:VectorSpace}} = nothing,
	name::Union{Symbol, String} = "A",
	IntrName::Union{Symbol, String} = string.(name),
) where L
	return addObs!(Tree, (Op,), (si,), (false,); pspace = pspace, name = (name,), IntrName = IntrName)
end

function addObs!(Tree::ObservableTree{L},
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


	return _insert_paired_channel!(Tree, Ops_idx, ref; phase = S.strength)
end
