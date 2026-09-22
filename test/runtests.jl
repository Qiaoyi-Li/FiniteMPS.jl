using Test
using FiniteMPS
using LinearAlgebra, Random

LinearAlgebra.BLAS.set_num_threads(1)

@testset "TensorKit interfaces" begin
     include("tensorkit_interfaces.jl")
     include("action_interfaces.jl")
     include("upgrade_sweep.jl")
end

@testset "Tree evaluation" begin
     include("TreeEval.jl")
end

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
     @testset "spinful2" verbose = true include("mulsiteIntr3.jl")
end
