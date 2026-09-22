"""
    calITP!(Tree::ImagTimeProxyTree, ρ::MPO; alg=LayeredTreeEval(),
        El=nothing, disk=false, maxsize=nothing, verbose=0, showtimes=10, GCspacing=0)

Evaluate the imaginary time proxies in `Tree` and write their values to `Tree.Refs`.
Return the accumulated `TimerOutput`. The execution and cache options have the same
meaning as in `calObs!`. A supplied `El` closes the caller's left auxiliary boundary;
otherwise a bilayer isometry is used. No observable normalization factor is applied.
"""
function calITP!(Tree::ImagTimeProxyTree{L}, ρ::MPO{L};
    alg::TreeEvalAlgorithm = LayeredTreeEval(), disk::Bool = false, maxsize = nothing,
    verbose::Integer = 0, showtimes = 10, GCspacing = 0, El = nothing, kwargs...) where L
    _tree_options(kwargs, maxsize, GCspacing, showtimes)
    merge!(Tree)
    left = if isnothing(El)
        boundary = codomain(ρ[1])[1]
        BilayerLeftTensor{1, 1}(isometry(boundary, boundary))
    else
        El
    end
    left = left isa AbstractTensorMap ? BilayerLeftTensor(left) : left
    lastsite = ρ[end]
    boundary = domain(lastsite, numin(lastsite))
    right = BilayerRightTensor{1, 1}(isometry(boundary, boundary))
    prepare(_, si) = (site = ρ[si]; (site', site))
    function pushenv(side, node, env, (bra, ket))
        A, B = Tree.Ops[node.Op[1]][node.Op[2]]
        allocator = ManualAllocator()
        return side ? _pushright(env, bra, A, ket, B; allocator) :
            _pushleft(env, bra, A, ket, B; allocator)
    end
    return _evaluate_tree!(Tree, prepare, pushenv, left, right, alg;
        disk, maxsize, verbose, showtimes, GCspacing)
end
