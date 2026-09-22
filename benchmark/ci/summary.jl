"""Bounded Actions summaries; detailed case data stays in the result artifact."""
module PerformanceSummary

using JSON3

export read_summary, write_summary

readjson(path) = JSON3.read(read(path, String), Dict{String, Any})

function read_summary(path::AbstractString)
    report = readjson(path)
    configurations = get(report, "configurations", nothing)
    if configurations isa AbstractVector
        first_report = readjson(joinpath(dirname(path), first(configurations)["report_path"]))
        count = length(first_report["cases"])
    else
        runtime = get(get(report, "environment", Dict()), "runtime", Dict())
        configurations = [Dict("julia_threads" => get(runtime, "julia_threads_default", nothing),
                               "blas_threads" => get(runtime, "blas_threads", nothing),
                               "gc_threads" => get(runtime, "julia_gc_threads", nothing))]
        count = length(report["cases"])
    end
    return (; source=get(report, "source", Dict()), run=get(report, "run", Dict()),
            configurations, case_count=count)
end

# Limit individual metadata values as well as the number of rows. Valid reports
# can contain arbitrarily long package versions/tags; these must not grow CI output.
function cell(value)
    value === nothing && return "unknown"
    text = string(value)
    text = length(text) > 200 ? first(text, 200) * "…" : text
    return "<code>" * replace(text, '&' => "&amp;", '<' => "&lt;", '>' => "&gt;",
        '|' => "&#124;", '\n' => " ", '\r' => " ") * "</code>"
end

valid_repository(value) = value isa AbstractString && ncodeunits(value) <= 256 &&
    occursin(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", value)
valid_run_url(value) = value isa AbstractString && ncodeunits(value) <= 2048 &&
    occursin(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/actions/runs/[0-9]+(?:/attempts/[0-9]+)?$", value)

function run_url(run, env)
    url = get(run, "run_url", nothing)
    valid_run_url(url) && return url
    repository, id = get(env, "GITHUB_REPOSITORY", nothing), get(env, "GITHUB_RUN_ID", nothing)
    valid_repository(repository) && id isa AbstractString &&
        occursin(r"^[0-9]{1,30}$", id) || return nothing
    return "https://github.com/$repository/actions/runs/$id"
end

url_segment(value) = join((byte in codeunits("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~") ?
    string(Char(byte)) : "%" * uppercase(string(byte; base=16, pad=2))) for byte in codeunits(value))

function write_summary(io::IO, metadata; publishing::AbstractString, git_push=nothing,
                       published_mode=nothing, env=ENV)
    measured = metadata !== nothing
    source, run = measured ? (metadata.source, metadata.run) : (Dict(), Dict())
    count = measured ? metadata.case_count : 0
    configurations = measured ? metadata.configurations : []
    println(io, "## Performance result\n\n| Field | Value |\n|---|---|")
    println(io, "| Status | ", measured ? "measured" : "empty — no cases configured", " |")
    println(io, "| Cases per configuration | $count |")
    println(io, "| Measured configurations | ", length(configurations), " |")
    println(io, "| Measurement points | ", count * length(configurations), " |")
    if measured
        for (label, value) in (("Repository", get(source, "repository", nothing)),
                ("Commit SHA", get(source, "commit_sha", nothing)),
                ("Tag", get(source, "tag", nothing)))
            println(io, "| $label | ", cell(value), " |")
        end
        println(io, "\n| Julia threads | BLAS threads | GC threads |\n|---|---|---|")
        for configuration in Iterators.take(configurations, 3)
            println(io, "| ", join((cell(get(configuration, key, nothing)) for key in
                ("julia_threads", "blas_threads", "gc_threads")), " | "), " |")
        end
    end
    println(io, "\nPublishing: ", cell(publishing), ".")
    git_push === nothing || println(io, "\nGit push: ", git_push ? "completed" : "no changes", ".")
    links = String[]
    repository, sha = get(source, "repository", nothing), get(source, "commit_sha", nothing)
    if valid_repository(repository)
        sha isa AbstractString && occursin(r"^[0-9a-fA-F]{40}$", sha) &&
            push!(links, "[Measured commit](https://github.com/$repository/commit/$sha)")
        tag = get(source, "tag", nothing)
        if published_mode == "dev"
            push!(links, "[Current published files](https://github.com/$repository/tree/gh-pages/performance/dev)")
        elseif published_mode == "release" && tag isa AbstractString && ncodeunits(tag) <= 256
            push!(links, "[Archived files](https://github.com/$repository/tree/gh-pages/performance/releases/$(url_segment(tag)))")
        end
    end
    url = run_url(run, env)
    url === nothing || push!(links, "[Workflow run](" * url * ")")
    isempty(links) || println(io, "\n", join(links, " · "))
    println(io, "\nComplete result files, including the full Markdown report, are retained by the result artifact step.\n")
    return nothing
end

end # module
