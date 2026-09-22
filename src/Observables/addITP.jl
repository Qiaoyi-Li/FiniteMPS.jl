"""
    addITP!(Tree::ImagTimeProxyTree, ((A1, ...), (B1, ...)), sites, fermionic;
        Z=nothing, pspace=nothing, name, ITPname, ValueType=Number)

Register an imaginary time proxy with two nonempty operator chains. The `sites`,
`fermionic` and `name` tuples list the first chain followed by the second. Each chain
is sorted, reduced and normalized to a closed right boundary independently; chains
open at both ends are unsupported. Results retain the original `sites` tuple as
the key. Repeated `(ITPname, sites)` registrations are ignored. The default name joins
the local names of each chain with an underscore between the chains.
"""
function addITP!(Tree::ImagTimeProxyTree{L}, ops::Tuple{Tuple, Tuple},
    sites::NTuple{N, Int}, fermionic::NTuple{N, Bool};
    Z::Union{Nothing, AbstractTensorMap, AbstractVector} = nothing,
    pspace::Union{Nothing, VectorSpace, AbstractVector{<:VectorSpace}} = nothing,
    name = _default_IntrName(N), ITPname = nothing, ValueType::Type{<:Number} = Number) where {L, N}
    m, n = length.(ops)
    m > 0 && n > 0 || throw(ArgumentError("both ITP operator chains must be nonempty"))
    m + n == N == length(name) || throw(ArgumentError("sites, fermionic flags and names must match the two chains"))
    all(si -> 1 <= si <= L, sites) || throw(ArgumentError("operator sites must lie in 1:$L"))
    any(fermionic) && isnothing(Z) && throw(ArgumentError("fermionic operators require Z"))
    names = string.(name)
    key = isnothing(ITPname) ? prod(names[1:m]) * "_" * prod(names[m+1:end]) : string(ITPname)
    haskey(Tree.Refs, key) && haskey(Tree.Refs[key], sites) && return nothing
    localops = _local_operator_chain((ops[1]..., ops[2]...), sites, fermionic, names)
    _validate_string(localops; rightclosed = true)
    A = StringOperator(collect(localops[1:m])) |> sort! |> reduce!
    B = StringOperator(collect(localops[m+1:end])) |> sort! |> reduce!
    phase = A.strength * B.strength
    paired = zip(_chain_iterator(A, Z, pspace, Val(L)), _chain_iterator(B, Z, pspace, Val(L)))
    indices = Int[]
    for (si, payload) in enumerate(paired)
        idx = findfirst(==(payload), Tree.Ops[si])
        if isnothing(idx)
            push!(Tree.Ops[si], payload)
            idx = length(Tree.Ops[si])
        end
        push!(indices, idx)
    end
    targets = get!(Tree.Refs, key) do
        Dict{NTuple{N, Int}, Ref{ValueType}}()
    end
    ref = Ref{ValueType}()
    _insert_paired_channel!(Tree, indices, ref; phase)
    targets[sites] = ref
    return nothing
end
