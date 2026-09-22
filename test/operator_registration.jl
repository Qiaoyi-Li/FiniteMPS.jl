@testset "String auxiliary channels" begin
    rng, V = MersenneTwister(81), ComplexSpace(2)
    A = randn(rng, ComplexF64, V ← V ⊗ V)
    B = randn(rng, ComplexF64, V ⊗ V ← V)
    P = LocalOperator(randn(rng, V ← V), "P", 2, false; aspace = (V, V))
    Q = LocalOperator(randn(rng, V ← V), "Q", 2, false; aspace = (V, V))
    S = StringOperator(LocalOperator(A, "A", 1, false), P, Q, LocalOperator(B, "B", 3, false))
    FiniteMPS.reduce!(S)
    @test length(S) == 3 && S[2].A ≈ P.A * Q.A && S[2].aspace == (V, V)

    p, bond = ComplexSpace(2), ComplexSpace(3)
    a, b, c = zeros(2, 2, 3), zeros(3, 2, 2, 3), zeros(3, 2, 2, 3)
    a[1, 1, 1] = a[2, 2, 2] = 1
    b[1, 1, 1, 1] = b[2, 2, 2, 2] = 1
    c[1, 1, 1, 1], c[2, 2, 2, 2] = 1, 1e-18
    A = TensorMap(a, p ← p ⊗ bond)
    B = TensorMap(b, bond ⊗ p ← p ⊗ bond)
    C = TensorMap(c, bond ⊗ p ← p ⊗ bond)
    S = StringOperator(LocalOperator(A, "A", 1, false), LocalOperator(B, "B", 2, false), LocalOperator(C, "C", 3, false))
    @tensor before[q a c e; b d f] := A[a b r] * B[r c d s] * C[s e f q]
    FiniteMPS._normalize_right_boundary!(S)
    @tensor after[q a c e; b d f] := S[1].A[q a b r] * S[2].A[r c d s] * S[3].A[s e f]
    @test before ≈ after && [dim(domain(S[i].A, 2)) for i in 1:2] == [1, 1]
end

@testset "Registered auxiliary boundaries" begin
    right = SU2Spin.SS[1]
    left = permute(right, ((3, 1), (2,)))
    for (Tree, register) in (
        (InteractionTree, (tree, op) -> addIntr!(tree, op, 2, 1.0; name = :S)),
        (ObservableTree, (tree, op) -> addObs!(tree, op, 2; name = :S)),
    )
        righttree, lefttree = Tree(2), Tree(2)
        register(righttree, right)
        register(lefttree, left)
        @test righttree.Ops == lefttree.Ops
    end
    q = domain(right, 2)
    bothopen = permute(id(q ⊗ SU2Spin.pspace), ((1, 2), (4, 3)))
    @test_throws ArgumentError addObs!(ObservableTree(2), bothopen, 1)
end
