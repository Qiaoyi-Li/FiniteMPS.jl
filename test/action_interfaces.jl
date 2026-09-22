@testset "action cache and prefusion" begin
    p, v = ℂ^2, ℂ^1
    D, X, Z = [1.0 0.0; 0.0 2.0], [0.0 1.0; 1.0 0.0], [0.5 0.0; 0.0 -0.5]
    density = TensorMap(D, p, p)
    site = permute(id(v) ⊗ density, ((1, 2), (4, 3)))
    rho = MPO([site, copy(site)])
    xop = TensorMap(X, p, p)
    zop = TensorMap(Z, p, p)
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
        effective = (0.7 * tr(D' * Z * D) - ph.E₀) * I - 0.4 * sum(abs2, D) * X
        dense = convert(Array, x.A)
        reference = TensorMap(reshape(effective * reshape(dense, 2, 2), size(dense)), space(x.A))
        @test action1(FiniteMPS._prefuse(ph), x).A ≈ reference
        @test action(x, cached).A ≈ reference
        buffers = [pointer(t.data) for term in cached.PH for t in term.cache]
        @test action(y, cached).A ≈ (-0.37 + 0.21im) * reference
        @test !isempty(buffers) && [pointer(t.data) for term in cached.PH for t in term.cache] == buffers
    finally
        finalize(cached)
    end
    @test all(isempty(term.cache) for term in cached.PH)
end
