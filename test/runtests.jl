using Test
using FiniteMPS

FiniteMPS.set_num_threads_mkl(1)

# @testset "replace TensorKit" verbose = true begin
#      include("replaced.jl")
# end

# @testset "complex support" verbose = true begin
#      include("complex.jl")
# end

@testset "ObsTree" verbose = true begin
     include("ObsTree.jl")
end

@testset "Automata MPO" verbose = true begin
     include("FreeFermion.jl")
end

# test multi-site interaction
@testset "Multi-site Intr" verbose = true begin
     @testset "spinless" verbose = true include("mulsiteIntr.jl")
     @testset "spinful" verbose = true include("mulsiteIntr2.jl")
end

