using Test, LinearAlgebra, Random

if ARGS == ["mkl"]
     include("mkl_threads.jl")
else
     BLAS.set_num_threads(2)
     using FiniteMPS

     @testset "Dependency contracts" begin
          @test BLAS.get_num_threads() == (FiniteMPS._is_mkl_backend() ? 2 : 1)
          @test (TensorKit.get_num_transformer_threads(), TensorKit.get_num_manipulation_threads(),
               TensorKit.Strided.get_num_threads(), TensorKit.timeit_debug_enabled()) == (1, 1, 1, false)
          BLAS.set_num_threads(1)
          include("tensorkit_interfaces.jl")
     end

     @testset "Composition regressions" begin
          include("operator_registration.jl")
          include("action_interfaces.jl")
          include("TreeEval.jl")
          include("ObsTree.jl")
          include("fermion_regressions.jl")
          include("upgrade_sweep.jl")
     end
end
