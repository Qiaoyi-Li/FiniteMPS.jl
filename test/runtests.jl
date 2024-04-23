using Test
using FiniteMPS

@testset "replace TensorKit" verbose = true begin
     include("replaced.jl")
end

@testset "complex support" verbose = true begin
     include("complex.jl")
end

@testset "ObsTree" verbose = true begin
     include("ObsTree.jl")
end