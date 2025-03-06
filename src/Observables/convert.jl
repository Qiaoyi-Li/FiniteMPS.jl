"""
	convert(T::Type, Tree::ObservableTree; kwargs...)

Collect the observables from the tree and store them in a dictionary or a named tuple. Current valid types are `Dict` and `NamedTuple`.  
"""
function convert(::Type{Dict}, Tree::ObservableTree; kwargs...)

	return Dict{String, Dict}(k => Dict{typeof(d).parameters[1], Number}(si => v[] for (si, v) in d) for (k, d) in Tree.Refs)
end
function convert(::Type{NamedTuple}, Tree::ObservableTree; kwargs...)
	Rslt = convert(Dict, Tree; kwargs...)
	k = keys(Rslt) .|> Symbol |> Tuple
	return NamedTuple{k}(values(Rslt))
end
