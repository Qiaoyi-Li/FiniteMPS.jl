@testset "SU2 sweep" begin
    tree = InteractionTree(4)
    for i in 1:3
        addIntr!(tree, SU2Spin.SS, (i, i + 1), (false, false), 1.0; name = (:S, :S))
    end
    ham = AutomataMPO(tree)
    psi = randMPS(MersenneTwister(83), 4, SU2Spin.pspace,
        Rep[SU₂](s => 1 for s in 0:1/2:1))
    env = Environment(psi', ham, psi)
    info, _ = DMRGSweep2!(env; K = 8, trunc = truncrank(8))
    @test info[2][1].Eg ≈ -(3 + 2sqrt(3)) / 4
end
