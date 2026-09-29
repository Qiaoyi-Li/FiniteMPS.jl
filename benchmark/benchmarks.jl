using FiniteMPS
include("models.jl")
include("fixtures.jl")
using .BenchmarkModels, .PerformanceFixtures

function sweep!(env, operation, D)
    options = (;K=4, trunc=truncrank(D), GCstep=false, GCsweep=false)
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
    for model in MODELS, kind in ("ground","thermal"), D in sort!(union(DIMENSIONS.two_site,DIMENSIONS.cbe))
        fixture = Ref{Any}(nothing)
        operations = String[]
        D in DIMENSIONS.two_site && push!(operations,kind == "ground" ? "2-DMRG" : "2-TDVP")
        D in DIMENSIONS.cbe && push!(operations,kind == "ground" ? "CBE-DMRG" : "CBE-TDVP")
        for (index, operation) in enumerate(operations)
            cbe = startswith(operation,"CBE-")
            params = Dict{String,Any}("model"=>model.id, "model_name"=>model.name,
                "symmetry"=>model.symmetry, "operation"=>operation, "state"=>kind,
                "nominal_D"=>D, "model_parameters"=>model_parameters(model,kind),
                "K"=>4, "truncation"=>"truncrank(D)", "scalar_type"=>"Float64",
                "GCstep"=>false, "GCsweep"=>false,
                "dt"=>kind == "thermal" ? -0.1 : nothing,
                "cbe_target"=>!cbe ? nothing : kind == "ground" ? 2D : D+div(D,8),
                "cbe_tolerance"=>cbe ? 1e-8 : nothing,
                "rsvd"=>cbe,
                "continuation_after"=>index == 1 ? nothing : first(operations),
                "sampling_state"=>"continue across warmup and measurement",
                "execution"=>Dict("julia_threads"=>config.julia_threads,"blas_threads"=>1,"gc_threads"=>1))
            build = rng -> begin
                index == 1 && (fixture[] = build_fixture(rng,model,kind,D))
                env = fixture[].env
                parameters = fixture[].parameters
                benchmark = @benchmarkable sweep!($env,$operation,$D)
                cleanup = index == length(operations) ? ()->(fixture[]=nothing) : ()->nothing
                (;benchmark, parameters, cleanup)
            end
            push!(cases, BenchmarkCase("$(model.id)/$(operation)/YC4x8/D$(D)/v2",build;
                parameters=params, seed=SEED, seconds=600.0, samples=1, evals=1, mutates=true,
                warmup_group="$(model.id)/$(operation)", warmup_size=D,
                description="One complete left-to-right and right-to-left $operation sweep."))
        end
    end
    return cases
end
