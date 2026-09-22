@testset "action cache and prefusion" begin
    p, v = ℂ^2, ℂ^1
    density = TensorMap([1.0 0.0; 0.0 2.0], p, p)
    site = permute(id(v) ⊗ density, ((1, 2), (4, 3)))
    rho = MPO([site, copy(site)])
    xop = TensorMap([0.0 1.0; 1.0 0.0], p, p)
    zop = TensorMap([0.5 0.0; 0.0 -0.5], p, p)
    tree = InteractionTree(2)
    addIntr!(tree, zop, 1, 0.7; name = :z)
    addIntr!(tree, xop, 2, -0.4; name = :x)
    ham = AutomataMPO(tree)
    env = Environment(rho', ham, rho)
    canonicalize!(env, 2)
    ph = ProjHam(env, 2; E₀ = 0.23)
    x = MPSTensor(randn(MersenneTwister(81), ComplexF64, space(rho[2].A)))
    y = (-0.37 + 0.21im) * x
    cached = CompositeProjectiveHamiltonian(env.El[2], env.Er[2], (ham[2],), ph.E₀)
    try
        reference = (action1(ph, x).A, action1(ph, y).A)
        observed = (action1(FiniteMPS._prefuse(ph), x).A, action(x, cached).A, action(y, cached).A)
        @test all(a ≈ b for (a, b) in zip(observed, (reference[1], reference...)))
    finally
        finalize(cached)
    end
    @test all(isempty(term.cache) for term in cached.PH)
end
