include("harness.jl")
using .BenchmarkHarness
import FiniteMPS

function main(args=ARGS)
    options = Dict{String,String}()
    allowed = ("--output", "--expected-sha", "--tag", "--prerelease", "--suite",
               "--julia-threads", "--blas-threads", "--gc-threads", "--samples", "--seconds")
    iseven(length(args)) || error("arguments must be --option value pairs")
    for i in 1:2:length(args)
        args[i] in allowed || error("unknown option: $(args[i])")
        haskey(options, args[i]) && error("duplicate option: $(args[i])")
        options[args[i]] = args[i + 1]
    end
    get(options, "--prerelease", "false") in ("true", "false") || error("--prerelease must be true or false")
    repo = dirname(@__DIR__)
    realpath(dirname(dirname(pathof(FiniteMPS)))) == realpath(repo) ||
        error("FiniteMPS must be loaded from this checkout; instantiate benchmark/Project.toml")
    run_benchmarks(; suite=get(options, "--suite", joinpath(@__DIR__, "benchmarks.jl")),
        output=get(options, "--output", joinpath(@__DIR__, "output")),
        expected_sha=get(options, "--expected-sha", nothing), tag=get(options, "--tag", nothing),
        prerelease=get(options, "--prerelease", "false") == "true",
        config=ExecutionConfig(parse(Int,get(options,"--julia-threads",string(Threads.nthreads(:default))));
            blas_threads=parse(Int,get(options,"--blas-threads","1")),
            gc_threads=parse(Int,get(options,"--gc-threads","1"))),
        samples=haskey(options,"--samples") ? parse(Int,options["--samples"]) : nothing,
        seconds=haskey(options,"--seconds") ? parse(Float64,options["--seconds"]) : nothing,
        packages=[FiniteMPS, FiniteMPS.TensorKit, FiniteMPS.TensorOperations,
                  FiniteMPS.MatrixAlgebraKit, BenchmarkHarness.BenchmarkTools],
        package_version=string(pkgversion(FiniteMPS)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
