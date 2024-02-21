# test if the TensorKit methods replaced are still correct

lsspace = [ℝ^2, ℂ^2,
     Rep[U₁](i => 2 for i in -1:1),
     Rep[SU₂](i => 2 for i in 0:1//2:1),
     Rep[U₁×SU₂]((i, j) => 2 for i in -1:1 for j in 0:1//2:1)]
lsstr_space = ["ℝ", "ℂ", "U₁", "SU₂", "U₁×SU₂"]
lsA = TensorMap[]
lsB = TensorMap[]
lsstr = String[]
for (s, str_space) in zip(lsspace, lsstr_space)
     if isa(s, CartesianSpace) 
          lsT = [Float64,]
     elseif isa(s, ComplexSpace)
          lsT = [ComplexF64,]
     else
          lsT = [Float64, ComplexF64]
     end
     for T in lsT
          push!(lsA, TensorMap(randn, T, s⊗s, s⊗s))
          push!(lsB, TensorMap(randn, T, s⊗s, s⊗s))
          push!(lsstr, "$(str_space)[$(T)]")
     end
end
FiniteMPS.set_num_threads_mul(1)

lsAB = map(lsA, lsB) do A, B 
     A*B
end

FiniteMPS.set_num_threads_mul(Threads.nthreads())
for (A, B, AB, str) in zip(lsA, lsB, lsAB, lsstr)
     @testset "$str" begin
          @test A*B ≈ AB
          for alg = [SVD(), SDD()]
               u, s, vd, ϵ = tsvd(AB; alg = alg)      
               @test u * s * vd ≈ AB
          end
          AA = A*A'
          D, V = eigh(AA)
          @test V * D * V' ≈ AA
          
     end
end



