function _tdvp_dense(state)
    tensor = ones(ComplexF64, 1, 1)
    for A in state
        site = convert(Array, A.A)
        tensor = reshape(tensor * reshape(site, size(site,1), :), :, size(site,ndims(site)))
    end
    tensor *= coef(state)
    state isa MPS && return vec(tensor)
    L = length(state)
    return reshape(permutedims(reshape(tensor, ntuple(_->2, 2L)),
        (collect(1:2:2L)..., collect(2:2:2L)...)), 2^L, 2^L)
end

@testset "Two-site TDVP MPS and MPO" begin
    rng = MersenneTwister(20260929)
    p = ℂ^2
    X = TensorMap([0.0 1.0; 1.0 0.0], p, p)
    Z = TensorMap([1.0 0.0; 0.0 -1.0], p, p)
    tree = InteractionTree(3)
    for i in 1:3
        addIntr!(tree, X, i, -0.2i; name=:X)
    end
    for i in 1:2
        addIntr!(tree, (Z,Z), (i,i+1), (false,false), 0.7; name=(:Z,:Z))
    end
    H = AutomataMPO(tree)
    Hdense = _dense_mpo(H)
    for ismpo in (false,true)
        bonds = ismpo ? (ℂ^1,ℂ^4,ℂ^4,ℂ^1) : (ℂ^1,ℂ^2,ℂ^2,ℂ^1)
        tensors = [randn(rng,ComplexF64,bonds[i]⊗p,
            ismpo ? p⊗bonds[i+1] : bonds[i+1]) for i in 1:3]
        initial = ismpo ? MPO(tensors) : MPS(tensors)
        canonicalize!(initial,3)
        canonicalize!(initial,1)
        normalize!(initial)
        dense = _tdvp_dense(initial)
        for dt in (-0.1, -0.03im)
            state = deepcopy(initial)
            env = Environment(state', H, state)
            TDVPSweep2!(env, dt; K=8, tol=1e-12, trunc=truncrank(64),
                E_shift=0.17, GCstep=false, GCsweep=false)
            @test _tdvp_dense(state) ≈ exp(dt*(Hdense - 0.17I))*dense atol=1e-9 rtol=1e-9
        end
        if ismpo
            info, _ = TDVPSweep2!(Environment(initial',H,initial), -0.1;
                K=2, trunc=truncrank(64), GCstep=false, GCsweep=false)
            @test maximum(step.Lanczos.numops for sweep in info for steps in (sweep.forward,sweep.backward) for step in steps) <= 2
        end
    end
end
