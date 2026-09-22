@testset "Product-state observables" begin
    θ = [0.17, 0.41, 0.83, 1.09]
    Ψ = MPS([TensorMap([cos(t), sin(t)], ℂ^1 ⊗ ℂ^2, ℂ^1) for t in θ])
    canonicalize!(Ψ, 1)
    Sz = TensorMap([0.5 0.0; 0.0 -0.5], ℂ^2, ℂ^2)
    sites = ((1,), (4,), (4, 1), (2, 2), (3, 1, 3), (4, 2, 1), (4, 2, 1, 3))
    tree = ObservableTree(length(θ))
    for indices in sites
        ops = ntuple(_ -> Sz, length(indices))
        addObs!(tree, ops, indices, ntuple(_ -> false, length(indices)); IntrName = Symbol("Sz", length(indices)))
    end
    calObs!(tree, Ψ)
    values = convert(NamedTuple, tree)
    for indices in sites
        reference = prod(1:length(θ)) do i
            power = count(==(i), indices)
            cos(θ[i])^2 * 0.5^power + sin(θ[i])^2 * (-0.5)^power
        end
        @test getproperty(values, Symbol("Sz", length(indices)))[indices] ≈ reference
    end
end
