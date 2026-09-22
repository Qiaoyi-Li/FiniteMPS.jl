using Test, Random, FiniteMPS

@testset "TensorKit decomposition interfaces" begin
    threads = (BLAS.get_num_threads(), TensorKit.Strided.get_num_threads())
    @test (TensorKit.get_num_transformer_threads(), TensorKit.get_num_manipulation_threads(), TensorKit.Strided.get_num_threads()) == (1, 1, 1)
    for V in (ComplexSpace(3), SU2Space(0 => 1, 1//2 => 1))
        A = id(Float64, V)
        U, S, Vh, err = FiniteMPS._raw_svd(A)
        @test S isa DiagonalTensorMap
        @test U * S * Vh ≈ A
        info = BondInfo(S)
        @test (info.D, info.DD) == bonddim(V)
        @test info.SE ≈ log(dim(V))
        U, S, Vh, err = FiniteMPS._raw_svd(A; trunc = truncrank(1))
        @test norm(U * S * Vh - A) ≈ err
    end

    rng = MersenneTwister(37)
    V = ComplexSpace(2)
    A = MPSTensor(randn(rng, ComplexF64, V ⊗ V ← V))
    original = copy(A.A)
    Q, R, _ = leftorth(A)
    @test Q * R ≈ original
    @test Q' * Q ≈ id(domain(Q))
    L, Q, _ = rightorth(A)
    @test L * Q ≈ permute(original, ((1,), (2, 3)))
    @test Q * Q' ≈ id(codomain(Q))
    @test A.A == original

    tag = ("bra", "ket")
    y = LocalLeftTensor(randn(rng, Float64, V ← V), tag)
    x = LocalLeftTensor(randn(rng, ComplexF64, V ← V), tag)
    original = copy(y.A)
    z = add!!(y, x, 2, 3)
    @test z.A ≈ 2 * x.A + 3 * original
    @test z.tag == tag
    @test y.A == original
    @test add!!(z, x) === z

    U = FiniteMPS._nearidentity(MersenneTwister(51), ComplexF64, V, 0.01)
    @test U' * U ≈ id(V)
    @test U == FiniteMPS._nearidentity(MersenneTwister(51), ComplexF64, V, 0.01)
    iso = TensorKit.randisometry(rng, Float64, V ⊗ V, V)
    @test iso' * iso ≈ id(V)
    @test_throws DimensionMismatch TensorKit.randisometry(rng, Float64, unitspace(V), V)
    memory = randMPS(MersenneTwister(73), 2, V, V)
    disk = randMPS(MersenneTwister(73), 2, V, V; disk = true)
    try
        @test disk isa MPS{2,Float64,StoreDisk} && coef(disk) == coef(memory) && all(disk[i].A == memory[i].A for i in 1:2)
    finally
        cleanup!(disk.A)
    end

    finite = Z3Space(0 => 2, 1 => 3, 2 => 1; dual = true)
    @test FiniteMPS._rsvd_trunc(ProductSpace(finite), 3) == Z3Space(0 => 1, 1 => 1, 2 => 2)
    target, source = Z3Space(0 => 2, 1 => 1), Z3Space(0 => 2, 1 => 1, 2 => 3)
    unit = unitspace(source)
    bra = MPSTensor(zeros(target ⊗ unit ← target))'
    ket = MPSTensor(zeros(source ⊗ unit ← source))
    El = FiniteMPS._simpleEl(bra, unit, ket)
    @test codomain(El) == ProductSpace(target) && domain(El) == unit ⊗ source && El * El' ≈ id(target)
    embeddings = oplusEmbed([target, target])
    @test embeddings[1] * embeddings[1]' + embeddings[2] * embeddings[2]' ≈ id(target ⊕ target)
    @test (BLAS.get_num_threads(), TensorKit.Strided.get_num_threads()) == threads
end

@testset "String auxiliary channels" begin
    rng, V = MersenneTwister(81), ComplexSpace(2)
    A = randn(rng, ComplexF64, V ← V ⊗ V)
    B = randn(rng, ComplexF64, V ⊗ V ← V)
    C = randn(rng, ComplexF64, V ← V ⊗ V)
    S = StringOperator(LocalOperator(A, "A", 1, false), LocalOperator(B, "B", 2, false), LocalOperator(C, "C", 3, false))
    @tensor before[q a c e; b d f] := A[a b r] * B[r c d] * C[e f q]
    FiniteMPS._normalize_right_boundary!(S)
    @tensor after[q a c e; b d f] := S[1].A[q a b r] * S[2].A[r c d s] * S[3].A[s e f]
    @test before ≈ after

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
