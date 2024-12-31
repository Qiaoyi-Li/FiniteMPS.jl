"""
	convert(T::Type, Tree::ObservableTree; kwargs...)

Collect the observables from the tree and store them in a dictionary or a named tuple. Current valid types are `Dict` and `NamedTuple`.  
"""
function convert(::Type{Dict}, Root::InteractionTreeNode; kwargs...)

	T = Dict{Tuple{Vararg{Int64}}, Number}
	Rslt = Dict{String, T}() # Tuple si => value

	for node in PreOrderDFS(Root)
		isnothing(node.Op) && continue
		node.Op.si == 0 && continue
		if !isnan(node.Op.strength)
			for (sites, name) in node.value
				if !haskey(Rslt, name)
					Rslt[name] = T()
				end
				Rslt[name][sites] = node.Op.strength
			end
		end
	end

	return Rslt
end
function convert(::Type{NamedTuple}, Root::InteractionTreeNode; kwargs...)

	Rslt = convert(Dict, Root; kwargs...)
	k = keys(Rslt) .|> Symbol |> Tuple
	return NamedTuple{k}(values(Rslt))
end

function convert(::Type{T}, Tree::ObservableTree; kwargs...) where T <: Union{NamedTuple, Dict}
	return convert(T, Tree.Root; kwargs...)
end
