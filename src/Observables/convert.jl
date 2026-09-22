"""
	convert(T::Type, Tree::ObservableTree; kwargs...)

Collect the observables from the tree and store them in a dictionary or a named tuple. Current valid types are `Dict` and `NamedTuple`.
"""
function convert(::Type{Dict}, Tree::ObservableTree; kwargs...)

	return Dict{String, Dict}(k => Dict{typeof(d).parameters[1], Number}(si => v[] for (si, v) in d) for (k, d) in Tree.Refs)
end

"""
	convert(T::Type, G::ImagTimeProxyTree; kwargs...)

Collect the observables from the tree `G` and store them in a dictionary or a named tuple. Current valid types are `Dict` and `NamedTuple`.
"""
function convert(::Type{Dict}, G::ImagTimeProxyTree; kwargs...)

	obs = Dict{String, Dict}()
	for (k, d) in G.Refs
		F = any(v -> isa(v[], Complex), values(d)) ? ComplexF64 : Float64
		obs[k] = Dict{typeof(d).parameters[1], F}(si => v[] for (si, v) in d)
	end
	return obs
end

function convert(::Type{NamedTuple}, Tree::Union{ObservableTree,ImagTimeProxyTree}; kwargs...)
	Rslt = convert(Dict, Tree; kwargs...)
	k = keys(Rslt) .|> Symbol |> Tuple
	return NamedTuple{k}(values(Rslt))
end

