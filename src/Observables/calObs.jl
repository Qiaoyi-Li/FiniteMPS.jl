"""
	calObs!(Tree::ObservableTree{L},
		Ψ::AbstractMPS{L},
		Φ::AbstractMPS{L} = Ψ;
		kwargs...
	) -> TO::TimerOutput

Calculate `⟨Ψ|O|Φ⟩` of all observables `O` stored in `Tree`, using bra `⟨Ψ|` and ket `|Φ⟩`.

# Kwargs
	El::AbstractTensorMap
	Er::AbstractTensorMap
Manually set boundary left or right environment tensor. Default is the `rank-2` isometry deduced from `Ψ` and `Φ`, which may be incorrect if the operators have nontrivial auxiliary spaces.

	normalize::Bool = false
If 'true', calculate `⟨Ψ|O|Φ⟩/(|Ψ||Φ|)` instead of `⟨Ψ|O|Φ⟩`.

	alg::TreeEvalAlgorithm = LayeredTreeEval()
Use the shared layered executor. Set `alg=LayeredTreeEval(ntasks=1)` for synchronous execution.

	disk::Bool = false
Store environment tensor in disk or not.

	maxsize = nothing
Maximum number of cached environments across both trees in disk mode. The default is the widest combined layer; zero disables the cache.

	verbose::Int64 = 0
Show the timer several times if `verbose > 0`.

	showtimes::Int64 = 10
Times to show the timer.

	GCspacing::Int64 = 0
A positive value requests full GC at stage barriers after this many completed pushes and joins.

"""
function calObs!(Tree::ObservableTree{L}, Ψ::AbstractMPS{L}, Φ::AbstractMPS{L} = Ψ;
    alg::TreeEvalAlgorithm = LayeredTreeEval(), disk::Bool = false, maxsize = nothing,
    verbose::Integer = 0, showtimes = 10, GCspacing = 0,
    normalize::Bool = false, El = nothing, Er = nothing, kwargs...) where L
    _tree_options(kwargs, maxsize, GCspacing, showtimes)
    merge!(Tree)
    left = isnothing(El) ? _defaultEl(Ψ, Φ) : El
    right = isnothing(Er) ? _defaultEr(Ψ, Φ) : Er
    left = left isa AbstractTensorMap ? LocalLeftTensor(left) : left
    right = right isa AbstractTensorMap ? LocalRightTensor(right) : right
    function prepare(_, si)
        ket = Φ[si]
        bra = Ψ === Φ ? ket' : Ψ[si]'
        return bra, ket
    end
    function pushenv(side, node, env, (bra, ket))
        op = deepcopy(Tree.Ops[node.Op[1]][node.Op[2]])
        op.strength[] = 1.0
        return side ? _pushright(env, bra, op, ket) : _pushleft(env, bra, op, ket)
    end
    timer = _evaluate_tree!(Tree, prepare, pushenv, left, right, alg;
        disk, maxsize, verbose, showtimes, GCspacing)
    if !normalize
        factor = coef(Ψ) * coef(Φ)
        for targets in values(Tree.Refs), ref in values(targets)
            ref[] *= factor
        end
    end
    return timer
end

function _defaultEl(Ψ::AbstractMPS{L}, Φ::AbstractMPS{L}) where L
	# default case, no horizontal bond
	firstΨ = Ψ[1]
	firstΦ = Ψ === Φ ? firstΨ : Φ[1]
	v1 = codomain(firstΨ, 1)
	v2 = codomain(firstΦ, 1)
	return v1 == v2 ? id(v1) : nothing
end
function _defaultEr(Ψ::AbstractMPS{L}, Φ::AbstractMPS{L}) where L
	# default case, no horizontal bond
	lastΦ = Φ[end]
	lastΨ = Ψ === Φ ? lastΦ : Ψ[end]
	v1 = domain(lastΦ, numin(lastΦ))
	v2 = domain(lastΨ, numin(lastΨ))
	return v1 == v2 ? id(v1) : nothing
end


