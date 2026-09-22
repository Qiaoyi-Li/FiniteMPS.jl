module BenchmarkHarness

using BenchmarkTools
using Dates
using LinearAlgebra
using Random
using JSON3

include("metadata.jl")
include("reports.jl")
using .PerformanceMetadata
using .PerformanceReports

export BenchmarkCase, ExecutionConfig, load_suite, measure_suite, run_benchmarks,
       validate_execution

"The requested measuring process configuration; verified against actual runtime values."
struct ExecutionConfig
    julia_threads::Int
    blas_threads::Int
    gc_threads::Int
    function ExecutionConfig(julia_threads::Integer=Threads.nthreads(:default);
                             blas_threads::Integer=1, gc_threads::Integer=1)
        julia_threads in (1, 2, 4) || throw(ArgumentError("Julia threads must be 1, 2, or 4"))
        blas_threads == 1 || throw(ArgumentError("BLAS threads must be 1"))
        gc_threads == 1 || throw(ArgumentError("GC threads must be 1"))
        new(Int(julia_threads), Int(blas_threads), Int(gc_threads))
    end
end

function validate_execution(config::ExecutionConfig)
    Threads.nthreads(:default) == config.julia_threads || error("actual Julia default threads differ from requested configuration")
    Threads.nthreads(:interactive) == 0 || error("benchmark processes must have zero interactive threads")
    Threads.ngcthreads() == config.gc_threads || error("run benchmark processes with --gcthreads=1")
    BLAS.get_num_threads() == config.blas_threads || error("actual BLAS threads differ from requested configuration")
    return config
end

json_parameters(value::NamedTuple) = Dict{String,Any}(string(k)=>json_parameters(v) for (k,v) in pairs(value))
json_parameters(value::AbstractDict) = Dict{String,Any}(string(k)=>json_parameters(v) for (k,v) in pairs(value))
json_parameters(value::AbstractVector) = [json_parameters(v) for v in value]
json_parameters(value) = value

"""
    BenchmarkCase(id, build; parameters=Dict(), seed=20260912,
                  seconds=1.0, samples=100, evals=1, mutates=false, description="",
                  warmup_group=id, warmup_size=0)

Register one stable workload. `build(rng)` returns
`(; benchmark = @benchmarkable(...))` with optional `parameters` and `cleanup`
fields. Constructed parameters add JSON metadata after inputs exist and cannot
replace declared keys. `cleanup()` releases the fixture even if timing fails.
The builder prepares inputs outside timing; the benchmark defines its own setup
and teardown. Mutating workloads must declare `mutates=true`, use `evals=1`, and
restore input state in BenchmarkTools' per-sample setup. Cases with the same
`warmup_group` share compiled methods. The selected case with the smallest
`warmup_size` runs first with one warmup sample; the remaining cases proceed
directly to measured samples. The default gives each case its own group.
Each case is built once, and warmup groups are local to one suite invocation.
The runner collects garbage after cleanup, before preparing the next case.
"""
struct BenchmarkCase{F}
    case_id::String
    build::F
    parameters::Dict{String,Any}
    seed::Int
    seconds::Float64
    samples::Int
    evals::Int
    mutates::Bool
    description::String
    warmup_group::String
    warmup_size::Int
end

function BenchmarkCase(id::AbstractString, build;
                       parameters=Dict(), seed::Integer=20260912,
                       seconds::Real=1.0, samples::Integer=100, evals::Integer=1,
                       mutates::Bool=false, description::AbstractString="",
                       warmup_group::AbstractString=id, warmup_size::Integer=0)
    occursin(r"^[A-Za-z0-9][A-Za-z0-9_.:/=+-]*$", id) ||
        throw(ArgumentError("case_id must be a nonempty stable identifier without whitespace"))
    isfinite(seconds) && seconds > 0 || throw(ArgumentError("seconds must be positive and finite"))
    samples >= 2 || throw(ArgumentError("samples must be at least 2"))
    evals >= 1 || throw(ArgumentError("evals must be positive"))
    mutates && evals != 1 && throw(ArgumentError("mutating cases require evals=1"))
    seed >= 0 || throw(ArgumentError("seed must be nonnegative"))
    isempty(strip(warmup_group)) && throw(ArgumentError("warmup_group must be nonempty"))
    warmup_size >= 0 || throw(ArgumentError("warmup_size must be nonnegative"))
    params = json_parameters(parameters)
    params isa AbstractDict || throw(ArgumentError("parameters must be an object"))
    validate_parameters(params)
    return BenchmarkCase(String(id), build, params, Int(seed), Float64(seconds),
                         Int(samples), Int(evals), mutates, String(description),
                         String(warmup_group), Int(warmup_size))
end

# Deliberately small JSON value contract: no tensors, callbacks or opaque objects
# may accidentally make it into the public report as workload parameters.
function validate_parameters(x)
    if x isa AbstractDict
        all(k -> k isa AbstractString, keys(x)) || throw(ArgumentError("parameter keys must be strings"))
        foreach(validate_parameters, values(x))
    elseif x isa AbstractVector
        foreach(validate_parameters, x)
    elseif x isa Real
        isfinite(x) || throw(ArgumentError("parameters must be finite"))
    elseif !(isnothing(x) || x isa AbstractString || x isa Bool)
        throw(ArgumentError("parameters must contain only JSON values"))
    end
    return x
end

function validate_cases(cases::AbstractVector)
    all(c -> c isa BenchmarkCase, cases) || throw(ArgumentError("suite must contain BenchmarkCase entries"))
    isempty(cases) && throw(ArgumentError("suite must not be empty"))
    ids = [c.case_id for c in cases]
    length(unique(ids)) == length(ids) || throw(ArgumentError("duplicate case_id in suite"))
    return cases
end

"Load benchmark_suite(config) from the selected suite."
function load_suite(path::AbstractString, config::ExecutionConfig=ExecutionConfig())
    isfile(path) || throw(ArgumentError("benchmark suite file does not exist"))
    scope = Module(gensym(:Workloads))
    Core.eval(scope, :(using BenchmarkTools, Random))
    Core.eval(scope, :(const BenchmarkCase = $BenchmarkCase))
    Core.eval(scope, :(const ExecutionConfig = $ExecutionConfig))
    # Allow suite authors to split workload definitions into ordinary Julia files.
    Core.eval(scope, :(include(path::AbstractString) = Base.include($scope, path)))
    Base.include(scope, abspath(path))
    isdefined(scope, :benchmark_suite) || throw(ArgumentError("suite must define benchmark_suite(config)"))
    factory = getfield(scope, :benchmark_suite)
    cases = Base.invokelatest(factory, config)
    cases isa AbstractVector || throw(ArgumentError("benchmark_suite(config) must return a vector"))
    return cases
end

function build_case(case::BenchmarkCase)
    Random.seed!(case.seed)
    built = Base.invokelatest(case.build, Xoshiro(case.seed))
    try
        built isa NamedTuple && haskey(built, :benchmark) ||
            throw(ArgumentError("$(case.case_id): build must return (; benchmark)"))
        built.benchmark isa BenchmarkTools.Benchmark ||
            throw(ArgumentError("$(case.case_id): benchmark must be a BenchmarkTools benchmark"))
        all(key->key in (:benchmark, :parameters, :cleanup), keys(built)) ||
            throw(ArgumentError("$(case.case_id): unsupported builder field"))
    catch
        built isa NamedTuple && haskey(built,:cleanup) && cleanup_fixture(built)
        rethrow()
    end
    return built
end

function built_parameters(case, built)
    extra = haskey(built, :parameters) ? json_parameters(built.parameters) : Dict{String,Any}()
    extra isa AbstractDict || throw(ArgumentError("$(case.case_id): built parameters must be an object"))
    validate_parameters(extra)
    isempty(intersect(keys(case.parameters), keys(extra))) ||
        throw(ArgumentError("$(case.case_id): built parameters conflict with declared parameters"))
    return merge(deepcopy(case.parameters), deepcopy(extra))
end

cleanup_fixture(built) = haskey(built, :cleanup) ? Base.invokelatest(built.cleanup) : nothing

"Schedule each group's smallest selected input before its other cases."
function measurement_plan(cases)
    representatives = Dict{String,BenchmarkCase}()
    for case in cases
        previous = get(representatives,case.warmup_group,nothing)
        if isnothing(previous) || case.warmup_size < previous.warmup_size
            representatives[case.warmup_group] = case
        end
    end
    scheduled = Set{String}()
    plan = NamedTuple{(:case,:warmup_case),Tuple{BenchmarkCase,BenchmarkCase}}[]
    for case in cases
        representative = representatives[case.warmup_group]
        if case.warmup_group ∉ scheduled
            push!(plan,(case=representative,warmup_case=representative))
            push!(scheduled,case.warmup_group)
        end
        case.case_id == representative.case_id || push!(plan,(case=case,warmup_case=representative))
    end
    return plan
end

"Measure selected cases; errors propagate and no partial report is returned."
function measure_suite(cases::AbstractVector; progress=stderr,
                       samples=nothing, seconds=nothing, timings=nothing)
    isnothing(samples) || samples >= 2 || throw(ArgumentError("samples override must be at least 2"))
    isnothing(seconds) || (isfinite(seconds) && seconds > 0) || throw(ArgumentError("seconds override must be positive and finite"))
    selected = validate_cases(cases)
    results = Dict{String,Any}[]
    for (case,warmup_case) in measurement_plan(selected)
        println(progress, "Preparing ", case.case_id)
        flush(progress)
        timing = Dict{String,Any}("case_id"=>case.case_id)
        started = time_ns()
        try
            push!(results, measure_case(case; progress, samples, seconds, timing, warmup_case))
        finally
            # The measuring call has returned, releasing its fixture references.
            timing["garbage_collection_seconds"] = @elapsed GC.gc()
            timing["total_seconds"] = (time_ns()-started)/1e9
            println(progress,"Completed ",case.case_id,": ",
                join(["$key=$(round(timing[key];digits=3))" for key in
                      ("preparation_seconds","warmup_seconds","sampling_seconds","cleanup_seconds","garbage_collection_seconds","total_seconds")
                      if haskey(timing,key)],", "))
            flush(progress)
        end
        isnothing(timings) || push!(timings,timing)
    end
    return results
end

function measure_case(case; progress, samples, seconds, timing=Dict{String,Any}(), warmup_case=case)
    built = nothing
    try
        timing["preparation_seconds"] = @elapsed begin
            built = build_case(case)
            parameters = built_parameters(case, built)
        end
        b = built.benchmark
        # Central budgets override macro defaults; no auto-tuning changes evals.
        b.params.seconds = something(seconds, case.seconds)
        b.params.samples = something(samples, case.samples)
        b.params.evals = case.evals
        b.params.overhead = 0.0
        b.params.gctrial = false
        b.params.gcsample = false
        println(progress, "Measuring ", case.case_id)
        flush(progress)
        Random.seed!(case.seed)
        warmup_samples = Int(case.case_id == warmup_case.case_id)
        timing["warmup_seconds"] = 0.0
        if warmup_samples == 1
            # Use the real evals for warmup too: setup semantics remain identical.
            timing["warmup_seconds"] = @elapsed Base.invokelatest(BenchmarkTools.run, b;
                samples=1, seconds=b.params.seconds, warmup=false)
        end
        Random.seed!(case.seed)
        timing["sampling_seconds"] = @elapsed trial = Base.invokelatest(BenchmarkTools.run, b; warmup=false)
        length(trial) >= 2 || error("$(case.case_id): fewer than two samples; increase seconds budget")
        estimate = BenchmarkTools.median(trial)
        return Dict(
            "case_id" => case.case_id,
            "description" => case.description,
            "parameters" => parameters,
            "measurement_parameters" => Dict(
                "seed" => case.seed, "evals" => trial.params.evals,
                "seconds_budget" => trial.params.seconds, "samples_budget" => trial.params.samples,
                "mutates" => case.mutates, "warmup_samples" => warmup_samples,
                "warmup_policy" => "smallest_size_per_group",
                "warmup_case_id" => warmup_case.case_id,
                "gc_after_case" => true,
                "gctrial" => trial.params.gctrial, "gcsample" => trial.params.gcsample,
                "overhead_ns" => trial.params.overhead),
            "samples" => length(trial), "median_time_ns" => estimate.time,
            "allocated_bytes" => estimate.memory, "allocations" => estimate.allocs)
    finally
        timing["cleanup_seconds"] = @elapsed isnothing(built) || cleanup_fixture(built)
    end
end

git_value(repo, args...) = strip(read(Cmd(["git", "-C", repo, args...]), String))
optional_env(name) = isempty(get(ENV, name, "")) ? nothing : ENV[name]
optional_int(name) = isnothing(optional_env(name)) ? nothing : parse(Int, ENV[name])

"Run one suite and write a complete report only after all selected cases succeed."
function run_benchmarks(; suite=joinpath(@__DIR__, "benchmarks.jl"),
                        output=joinpath(@__DIR__, "output"), repo=dirname(@__DIR__),
                        expected_sha=nothing, tag=nothing, prerelease=false,
                        packages=Module[], package_version=nothing,
                        config::ExecutionConfig=ExecutionConfig(), samples=nothing, seconds=nothing)
    started = time_ns()
    # A reusable output directory must not leave stale success artifacts on error.
    output = abspath(output)
    mkpath(output)
    for name in ("report.json", "report.md", "run-status.json")
        path = joinpath(output, name)
        ispath(path) && rm(path)
    end
    sha = git_value(repo, "rev-parse", "HEAD")
    isnothing(expected_sha) || sha == expected_sha || error("checkout SHA differs from expected target")
    suite_relative = relpath(isfile(suite) ? realpath(suite) : abspath(suite), realpath(repo))
    suite_external = suite_relative == ".." || startswith(suite_relative, "../")
    dirty = suite_external || !isempty(git_value(repo, "status", "--porcelain", "--untracked-files=normal"))
    in_ci = get(ENV, "GITHUB_ACTIONS", "false") == "true"
    in_ci && dirty && error("CI checkout has modifications or an external suite")
    in_ci && realpath(suite) != realpath(joinpath(@__DIR__, "benchmarks.jl")) &&
        error("CI measurements must use the registered benchmark/benchmarks.jl suite")
    BLAS.set_num_threads(config.blas_threads)
    cases = load_suite(suite, config)
    # Suite imports may initialize a BLAS provider.
    BLAS.set_num_threads(config.blas_threads)
    validate_execution(config)
    if isempty(cases)
        write(joinpath(output, "report.md"), "# Performance\n\nNo benchmark cases configured. " *
              "The infrastructure is ready; no performance report was measured or published.\n\n" *
              "Checkout: `$sha`\n")
        PerformanceReports.write_json(joinpath(output, "run-status.json"),
                                      Dict("status" => "empty", "case_count" => 0))
        return nothing
    end
    validate_cases(cases)
    package_version isa AbstractString && !isempty(package_version) ||
        throw(ArgumentError("package_version is required for a nonempty suite; run.jl supplies the loaded package version"))
    isnothing(tag) || PerformanceReports.version_tag(tag)
    environment = PerformanceMetadata.collect_environment(; packages)
    measured_at = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sssZ")
    timings = Dict{String,Any}[]
    runner_setup_seconds = (time_ns()-started)/1e9
    sampling_seconds = @elapsed results = measure_suite(cases; samples, seconds, timings)
    validate_execution(config)
    for package in packages
        version = Base.pkgversion(package)
        environment["package_versions"][string(nameof(package))] = isnothing(version) ? nothing : string(version)
    end
    repository = something(optional_env("GITHUB_REPOSITORY"), "Qiaoyi-Li/FiniteMPS.jl")
    run_id = optional_env("GITHUB_RUN_ID")
    report = Dict{String,Any}(
        "schema_version" => 1,
        "source" => Dict("repository" => repository, "commit_sha" => sha, "tag" => tag,
                         "package_version" => package_version, "benchmark_source_sha" => sha,
                         "prerelease" => prerelease, "working_tree_dirty" => dirty,
                         "suite_path" => suite_external ? nothing : suite_relative,
                         "suite_origin" => suite_external ? "external" : "checkout"),
        "run" => Dict("measured_at_utc" => measured_at, "event_name" => something(optional_env("GITHUB_EVENT_NAME"), "local"),
                      "workflow" => optional_env("GITHUB_WORKFLOW"),
                      "run_id" => run_id, "run_number" => optional_int("GITHUB_RUN_NUMBER"),
                      "run_attempt" => optional_int("GITHUB_RUN_ATTEMPT"),
                      "run_url" => isnothing(run_id) ? nothing : "https://github.com/$repository/actions/runs/$run_id"),
        "environment" => environment, "cases" => results,
        "execution_timings" => Dict("runner_setup_seconds"=>runner_setup_seconds,
            "suite_seconds"=>sampling_seconds,"cases"=>timings))
    PerformanceReports.validate_report(report)
    markdown = PerformanceReports.render_markdown(report)
    PerformanceReports.write_json(joinpath(output, "report.json"), report)
    write(joinpath(output, "report.md"), markdown)
    PerformanceReports.write_json(joinpath(output, "run-status.json"),
                                  Dict("status" => "measured", "case_count" => length(results)))
    return report
end

end # module
