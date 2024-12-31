# use free fermion DMRG to test if the AutomataMPO works correctly
using LinearAlgebra: eigvals

function _free_fermion_U1DMRG(Tij::Matrix{T}, a::Int64) where {T<:Union{Float64,ComplexF64}}
     @assert a in [1, 2] # 2 ways to add terms
     @assert ishermitian(Tij)
     L = size(Tij, 1)

     # H = - T_{i,j} c_i^dag c_j
     Root = InteractionTreeNode()
     if a == 1
          for i in 1:L, j in 1:L
               i == j && continue
               addIntr2!(Root, U1SpinlessFermion.FdagF, (i, j),
                    -Tij[i, j]; Z=U1SpinlessFermion.Z, name=(:Fdag, :F))
          end
     elseif a == 2
          for i in 1:L, j in i+1:L
               addIntr2!(Root, U1SpinlessFermion.FdagF, (i, j),
                    -Tij[i, j]; Z=U1SpinlessFermion.Z, name=(:Fdag, :F))
               addIntr2!(Root, U1SpinlessFermion.FFdag, (i, j),
                    Tij[i, j]'; Z=U1SpinlessFermion.Z, name=(:F, :Fdag))
          end
     end

     for i in 1:L
          addIntr1!(Root, U1SpinlessFermion.n, i, -Tij[i, i]; name=:n)
     end

     H = AutomataMPO(Root)
     Ψ = randMPS(T, L, U1SpinlessFermion.pspace, Rep[U₁](c => 1 for c in -1:1/2:1))

     Env = Environment(Ψ', H, Ψ)
     Eg = 0.0
     for _ in 1:10
          info, _ = DMRGSweep2!(Env; trunc=truncdim(32))
          Eg = info[2][1].Eg
     end

     return Eg

end

function _free_fermion_U1U1DMRG(T₊::Matrix{T}, T₋::Matrix{T}, a::Int64) where {T<:Union{Float64,ComplexF64}}
     @assert a in [1, 2] # 2 ways to add terms
     @assert ishermitian(T₊) && ishermitian(T₋)
     L = size(T₊, 1)

     # H = - T_{i,j} c_i^dag c_j
     Root = InteractionTreeNode()
     if a == 1
          for i in 1:L, j in 1:L
               i == j && continue
               addIntr2!(Root, U1U1Fermion.FdagF₊, (i, j),
                    -T₊[i, j]; Z=U1U1Fermion.Z, name=(:Fdag₊, :F₊))
               addIntr2!(Root, U1U1Fermion.FdagF₋, (i, j),
                    -T₋[i, j]; Z=U1U1Fermion.Z, name=(:Fdag₋, :F₋))
          end
     elseif a == 2
          for i in 1:L, j in i+1:L
               addIntr2!(Root, U1U1Fermion.FdagF₊, (i, j),
                    -T₊[i, j]; Z=U1U1Fermion.Z, name=(:Fdag₊, :F₊))
               addIntr2!(Root, U1U1Fermion.FdagF₋, (i, j),
                    -T₋[i, j]; Z=U1U1Fermion.Z, name=(:Fdag₋, :F₋))


               addIntr2!(Root, U1U1Fermion.FFdag₊, (i, j),
                    T₊[i, j]'; Z=U1U1Fermion.Z, name=(:F₊, :Fdag₊))
               addIntr2!(Root, U1U1Fermion.FFdag₋, (i, j),
                    T₋[i, j]'; Z=U1U1Fermion.Z, name=(:F₋, :Fdag₋))
          end
     end

     for i in 1:L
          O = T₊[i, i] * U1U1Fermion.n₊ + T₋[i, i] * U1U1Fermion.n₋
          addIntr1!(Root, O, i, -1.0; name=:O)
     end

     H = AutomataMPO(Root)
     Ψ = randMPS(T, L, U1U1Fermion.pspace, Rep[U₁×U₁]((c, s) => 1 for c in -1:1:1 for s in -1:1/2:1))

     Env = Environment(Ψ', H, Ψ)
     Eg = 0.0
     for _ in 1:10
          info, _ = DMRGSweep2!(Env; trunc=truncdim(128))
          Eg = info[2][1].Eg
     end

     return Eg

end

function _GS_free_fermion(Tij::Matrix)
     @assert ishermitian(Tij)
     L = size(Tij, 1)
     @assert iseven(L)

     ϵ = eigvals(-Tij)
     # half filled
     return sum(ϵ[1:div(L, 2)])
end

function _GS_free_fermion(T₊::Matrix, T₋::Matrix)
     @assert ishermitian(T₊) && ishermitian(T₋)
     L = size(T₊, 1)
   
     Tij = cat(T₊, T₋; dims = (1,2))

     ϵ = eigvals(-Tij)
     # half filled
     return sum(ϵ[1:L])
end

L = 6
@testset "free fermion" verbose = true begin
     @testset "spinless(real)" begin
          Tij = rand(L, L)
          Tij += Tij'
          Eg = _GS_free_fermion(Tij)

          @test _free_fermion_U1DMRG(Tij, 1) ≈ Eg
          @test _free_fermion_U1DMRG(Tij, 2) ≈ Eg
     end

     @testset "spinless(complex)" begin
          Tij = rand(ComplexF64, L, L)
          Tij += Tij'
          Eg = _GS_free_fermion(Tij)

          @test _free_fermion_U1DMRG(Tij, 1) ≈ Eg
          @test _free_fermion_U1DMRG(Tij, 2) ≈ Eg
     end

     @testset "spinful(real)" begin
          T₊ = rand(L, L)
          T₊ += T₊'
          T₋ = rand(L, L)
          T₋ += T₋'
          Eg = _GS_free_fermion(T₊, T₋)

          @test _free_fermion_U1U1DMRG(T₊, T₋, 1) ≈ Eg
          @test _free_fermion_U1U1DMRG(T₊, T₋, 2) ≈ Eg
     end

     @testset "spinful(complex)" begin
          T₊ = rand(ComplexF64, L, L)
          T₊ += T₊'
          T₋ = rand(ComplexF64, L, L)
          T₋ += T₋'
          Eg = _GS_free_fermion(T₊, T₋)

          @test _free_fermion_U1U1DMRG(T₊, T₋, 1) ≈ Eg
          @test _free_fermion_U1U1DMRG(T₊, T₋, 2) ≈ Eg
     end

end