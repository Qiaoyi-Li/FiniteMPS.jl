using Test, Random, FiniteMPS

@testset "TensorKit decomposition interfaces" begin
    V = SU2Space(0 => 2, 1//2 => 1)
    A = zeros(Float64, V, V)
    for (c, b) in blocks(A)
        b .= dim(c) == 1 ? [4.0 0.0; 0.0 2.0] : ones(1, 1)
    end
    U, S, Vh, _ = FiniteMPS._raw_svd(A)
    @test S isa DiagonalTensorMap && U * S * Vh ≈ A
    info, probabilities = BondInfo(S), [16, 4, 1, 1] / 22
    @test (info.D, info.DD) == bonddim(V) == (3, 4) && info.SE ≈ -sum(p -> p * log(p), probabilities)
    shared = map(collect(blocks(S))) do (c, b)
        values = FiniteMPS.MatrixAlgebraKit.diagview(b)
        values[1] = -7
        block(S, c)[1, 1] == -7
    end
    @test all(shared)
    U, S, Vh, info = tsvd(MPSTensor(A), ((1,), (2,)); trunc = truncrank(1), p = 2,
        alg = FiniteMPS.MatrixAlgebraKit.QRIteration(fixgauge = false),
        CBEAlg = NaiveCBE(2, 1e-8), tol = 1e-12)
    @test info.TrunErr ≈ sqrt(6) && norm(U * S * Vh - A) ≈ info.TrunErr
    @test_throws ArgumentError tsvd(MPSTensor(A), ((1,), (2,)); p = 1)

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
    @test z.A ≈ 2 * x.A + 3 * original && z.tag == tag && y.A == original
    original = copy(z.A)
    @test add!!(z, x) === z && z.A ≈ original + x.A

    U = FiniteMPS._nearidentity(MersenneTwister(51), ComplexF64, V, 0.01)
    @test U' * U ≈ id(V)
    memory = randMPS(MersenneTwister(73), 2, V, V)
    disk = randMPS(MersenneTwister(73), 2, V, V; disk = true)
    try
        @test disk isa MPS{2,Float64,StoreDisk} && coef(disk) == coef(memory) && all(disk[i].A == memory[i].A for i in 1:2)
    finally
        cleanup!(disk.A)
    end

    finite = Z3Space(0 => 2, 1 => 3, 2 => 1; dual = true)
    @test FiniteMPS._rsvd_trunc(ProductSpace(finite), 3) == Z3Space(0 => 1, 1 => 1, 2 => 2)
    target, source = Z3Space(0 => 3, 1 => 1), Z3Space(0 => 2, 1 => 1, 2 => 3)
    unit = unitspace(source)
    bra = MPSTensor(zeros(target ⊗ unit ← target))'
    ket = MPSTensor(zeros(source ⊗ unit ← source))
    El = FiniteMPS._simpleEl(bra, unit, ket)
    c = one(sectortype(target))
    @test codomain(El) == ProductSpace(target) && domain(El) == unit ⊗ source && block(El, c) == [1 0; 0 1; 0 0]
    embeddings = oplusEmbed([Z3Space(0 => 1, 1 => 2), Z3Space(0 => 2, 2 => 1)])
    @test block(embeddings[1], c) == reshape([1, 0, 0], 3, 1) && block(embeddings[2], c) == [0 0; 1 0; 0 1]
end
