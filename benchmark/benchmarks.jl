using FiniteMPS
include("models.jl")
include("fixtures.jl")
using .BenchmarkModels, .PerformanceFixtures

function sweep!(env, operation, D)
    options = (;K=8, trunc=truncrank(D), GCstep=false, GCsweep=false)
    if operation == "2-DMRG"
        return DMRGSweep2!(env; options...)
    elseif operation == "CBE-DMRG"
        return DMRGSweep1!(env; options..., CBEAlg=NaiveCBE(2D,1e-8;rsvd=true))
    elseif operation == "2-TDVP"
        return TDVPSweep2!(env,-0.1; options...)
    else
        return TDVPSweep1!(env,-0.1; options..., CBEAlg=NaiveCBE(D+div(D,8),1e-8;rsvd=true))
    end
end

function benchmark_suite(config)
    FiniteMPS.set_num_threads_action(config.julia_threads)
    cases = BenchmarkCase[]
    for model in MODELS, kind in ("ground","thermal"), D in DIMENSIONS
        # Both algorithms and their samples continue the same state/environment.
        fixture = Ref{Any}(nothing)
        operations = kind == "ground" ? ("2-DMRG","CBE-DMRG") : ("2-TDVP","CBE-TDVP")
        for (index, operation) in enumerate(operations)
            params = Dict{String,Any}("model"=>model.id, "model_name"=>model.name,
                "symmetry"=>model.symmetry, "operation"=>operation, "state"=>kind,
                "nominal_D"=>D, "model_parameters"=>model_parameters(model,kind),
                "K"=>8, "truncation"=>"truncrank(D)", "scalar_type"=>"Float64",
                "GCstep"=>false, "GCsweep"=>false,
                "dt"=>kind == "thermal" ? -0.1 : nothing,
                "cbe_target"=>index == 1 ? nothing : kind == "ground" ? 2D : D+div(D,8),
                "cbe_tolerance"=>index == 1 ? nothing : 1e-8,
                "rsvd"=>index == 2,
                "continuation_after"=>index == 1 ? nothing : first(operations),
                "sampling_state"=>"continue across warmup, samples and paired algorithms",
                "execution"=>Dict("julia_threads"=>config.julia_threads,"blas_threads"=>1,"gc_threads"=>1))
            build = rng -> begin
                index == 1 && (fixture[] = build_fixture(rng,model,kind,D))
                env = fixture[].env
                parameters = fixture[].parameters
                benchmark = @benchmarkable sweep!($env,$operation,$D)
                cleanup = index == 2 ? ()->(fixture[]=nothing) : ()->nothing
                (;benchmark, parameters, cleanup)
            end
            push!(cases, BenchmarkCase("$(model.id)/$(operation)/YC4x8/D$(D)/v1",build;
                parameters=params, seed=SEED, seconds=600.0, samples=3, evals=1, mutates=true,
                warmup_group="$(model.id)/$(operation)", warmup_size=D,
                description="One complete left-to-right and right-to-left $operation sweep. Samples continue the same state; the CBE algorithm follows its two-site partner."))
        end
    end
    return cases
end
