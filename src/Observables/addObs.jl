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

Add an `N`-site observable to `Tree`, where the observable is characterized by `N`-tuples `Op`, `si` and `fermionic`. The implementation and usage is similar to `addIntr!`, except for the logic to deal with same operator which is added twice.

# Kwargs 
	Z::Union{Nothing, AbstractTensorMap, Vector{<:AbstractTensorMap}}
Provide the parity operator to deal with the fermionic anti-commutation relations.If `Z == nothing`, assume all operators are bosonic. Otherwise, a uniform (single operator) `Z::AbstractTensorMap` or site-dependent (length `L` vector) `Z::Vector{<:AbstractTensorMap}` should be given.

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
	Z::Union{Nothing, AbstractTensorMap, Vector{<:AbstractTensorMap}} = nothing,
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

	# construct LocalOperators
	ref_aspace = Ref{VectorSpace}(trivial(codomain(Op[1])[1]))
	lsOp = map(Op, si, fermionic, name) do o, s, f, n
		Oi = LocalOperator(o, n, s, f; aspace = (ref_aspace[], ref_aspace[])) 
		ref_aspace[] =  getRightSpace(Oi)
		return Oi
	end

	# make sure the auxiliary bond is on the left 
	lsOp = _rightOp(lsOp)

	S = StringOperator(lsOp..., 1.0) |> sort! |> reduce!
     # move the possible coefficient -1 to the last operator
     S.Ops[end].A = S.strength * S.Ops[end].A
     S.strength = 1.0

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
