using Test
using FiniteMPS

@testset "replace TensorKit" verbose = true begin
     include("replaced.jl")
end