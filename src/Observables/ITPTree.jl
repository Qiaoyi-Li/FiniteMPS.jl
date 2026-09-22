"""
    ImagTimeProxyTree(L)

Store imaginary time proxies in a shared prefix/suffix tree. Register terms with
`addITP!` and evaluate them with `calITP!`. Results are stored in `Refs`.
"""
mutable struct ImagTimeProxyTree{L}
    Ops::Vector{Vector{NTuple{2, AbstractLocalOperator}}}
    Refs::Dict{String, Dict}
    RootL::InteractionTreeNode
    RootR::InteractionTreeNode
    function ImagTimeProxyTree(L::Int)
        L > 0 || throw(ArgumentError("the number of sites must be positive"))
        return new{L}([NTuple{2, AbstractLocalOperator}[] for _ in 1:L],
            Dict{String, Dict}(), InteractionTreeNode((0, 0), nothing),
            InteractionTreeNode((L + 1, 0), nothing))
    end
end

"""
    merge!(Tree::ImagTimeProxyTree)

Merge shared prefixes and suffixes while retaining all result references.
"""
merge!(Tree::ImagTimeProxyTree) = _merge_paired!(Tree, Val(false))
