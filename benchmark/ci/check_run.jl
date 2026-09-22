"""Validate the runner completion contract before making an artifact publishable."""
module PerformanceCompletion

using JSON3
include(joinpath(@__DIR__,"..","collection_reports.jl"))
using .PerformanceCollections
include("summary.jl")
using .PerformanceSummary

export check_run

function check_run(directory::AbstractString)::Bool
    status = JSON3.read(read(joinpath(directory, "run-status.json"), String), Dict{String, Any})
    count = get(status, "case_count", nothing)
    count isa Integer && !(count isa Bool) && count >= 0 ||
        error("run-status.json must contain a nonnegative integer case_count")
    isfile(joinpath(directory, "report.md")) || error("Completed run must produce report.md")
    measured = get(status, "status", nothing) == "measured"
    report_path = joinpath(directory, "report.json")
    collection_path = joinpath(directory,"collection.json")
    if measured && isfile(collection_path)
        count > 0 && !ispath(report_path) || error("Collection requires cases and no ambiguous top-level report.json")
        get(status,"configuration_count",nothing)==3 || error("Collection must complete three configurations")
        bundle=read_collection(collection_path)
        length(first(bundle.reports)["cases"])==count || error("Collection case count differs from completion status")
        isfile(joinpath(directory,"index.html")) || error("Completed collection requires index.html")
    elseif measured
        count > 0 && isfile(report_path) || error("Measured run requires cases and report.json")
        report = JSON3.read(read(report_path, String), Dict{String, Any})
        schema = get(report, "schema_version", nothing)
        cases = get(report, "cases", nothing)
        schema isa Integer && !(schema isa Bool) && schema == 1 &&
            cases isa AbstractVector && length(cases) == count ||
            error("Measured report must be schema v1 and match the completed case count")
    elseif get(status, "status", nothing) != "empty" || count != 0 || ispath(report_path) || islink(report_path) ||
           ispath(collection_path) || islink(collection_path)
        error("Empty run requires zero cases and no report.json")
    end
    return measured
end

function main(args=ARGS)
    directory = mode = nothing
    index = 1
    while index <= length(args)
        if args[index] == "--mode"
            mode === nothing && index < length(args) || error("Missing or duplicate --mode")
            index += 1
            mode = args[index]
        else
            directory === nothing && !startswith(args[index], "--") || error("Unexpected argument: $(args[index])")
            directory = args[index]
        end
        index += 1
    end
    directory !== nothing && mode in ("none", "dev", "release") ||
        error("Usage: check_run.jl DIRECTORY --mode none|dev|release")
    measured = check_run(directory)
    open(ENV["GITHUB_OUTPUT"], "a") do stream
        println(stream, "measured=", measured)
    end
    open(ENV["GITHUB_STEP_SUMMARY"], "a") do stream
        input = isfile(joinpath(directory, "collection.json")) ? "collection.json" : "report.json"
        metadata = measured ? read_summary(joinpath(directory, input)) : nothing
        publishing = !measured ? "skipped — no cases configured; no performance data was published" :
            mode == "none" ? "disabled for manual run-only" : "pending — see the publish job for the final status"
        write_summary(stream, metadata; publishing)
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PerformanceCompletion.main()
end
