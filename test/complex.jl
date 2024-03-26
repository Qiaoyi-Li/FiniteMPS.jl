# test potential errors when there exist complex operators

Tree = InteractionTreeNode()
for i in 1:8
     if i < 8
     addIntr!(Tree, (NoSymSpinOneHalf.Sy, NoSymSpinOneHalf.Sy),
          (i, i+1), 1.0)
     end
     addIntr!(Tree, NoSymSpinOneHalf.Sz, i, 1.0)
end
H = AutomataMPO(Tree)


@testset "MPO" begin
     # test SETTN
     ρ, lsF = SETTN(H, 2^(-5); trunc = truncdim(8))
     @test scalartype(ρ) == ComplexF64

     # TDVP
     Env = Environment(ρ', H, ρ)
     dt = rand(ComplexF64)
     @test begin
          TDVPSweep2!(Env, dt; trunc = truncdim(8))
          true
     end
     @test begin
          TDVPSweep1!(Env, dt; trunc = truncdim(8),
               CBEAlg = CheapCBE(8, 1e-8))
          true
     end
     @test begin
          TDVPSweep1!(Env, dt)
          true
     end

end 

@testset "MPS" begin
     Ψ = randMPS(ComplexF64, length(H), NoSymSpinOneHalf.pspace, ℂ^2) 

     Env = Environment(Ψ', H, Ψ)
     @test begin 
          DMRGSweep2!(Env; trunc = truncdim(4))
          true
     end
     @test begin
          DMRGSweep1!(Env; trunc = truncdim(4),
               CBEAlg = CheapCBE(4, 1e-8))
          true
     end
     @test begin
          DMRGSweep1!(Env)
          true
     end


     # TDVP
     normalize!(Ψ)
     dt = im * rand()
     @test begin
          TDVPSweep2!(Env, dt; trunc = truncdim(8))
          norm(Ψ) ≈ 1
     end
     @test begin
          TDVPSweep1!(Env, dt; trunc = truncdim(8))
          norm(Ψ) ≈ 1
     end
     @test begin
          TDVPSweep1!(Env, dt; trunc = truncdim(8),
               CBEAlg = CheapCBE(4, 1e-8))
          norm(Ψ) ≈ 1
     end
end