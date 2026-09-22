module PerformanceReports

using JSON3
using Dates
include("workload_labels.jl")
using .WorkloadLabels

export validate_report, read_report, write_json, render_markdown, render_report, html_escape,
       page, json_text, version_tag, workload_title,
       workload_description, workload_facts,
       metadata_rows, measurement_rows, display_value, time_value, case_name, package_rows,
       ordered_cases

required(object, key, context) = haskey(object, key) ? object[key] : error("Missing $context.$key")
check(condition, message) = condition || error(message)
isobject(x) = x isa AbstractDict
isnumber(x) = x isa Real && !(x isa Bool) && isfinite(x) && x >= 0
iscount(x; positive=false) = x isa Integer && !(x isa Bool) && (positive ? x > 0 : x >= 0)
istext(x) = x isa AbstractString && !isempty(x)
optional(x, predicate) = x === nothing || predicate(x)

"""Parse only complete vMAJOR.MINOR.PATCH SemVer tags, with optional prerelease/build."""
function version_tag(tag)
    tag isa AbstractString || error("Version tag must be a string")
    pattern = r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
    match_result = match(pattern, tag)
    match_result === nothing && error("Invalid version tag: $tag")
    prerelease = match_result.captures[4]
    if prerelease !== nothing
        for identifier in split(prerelease, '.')
            all(isdigit, identifier) && length(identifier) > 1 && startswith(identifier, "0") &&
                error("Numeric prerelease identifiers cannot have leading zeros")
        end
    end
    return VersionNumber(tag[2:end])
end

"""Validate schema v1 without querying the publishing process or machine."""
function validate_report(report)
    check(isobject(report), "Report must be a JSON object")
    check(required(report, "schema_version", "report") === 1, "Unsupported report schema")
    source = required(report, "source", "report")
    run = required(report, "run", "report")
    environment = required(report, "environment", "report")
    for (name, object) in (("source", source), ("run", run), ("environment", environment))
        check(isobject(object), "$name must be an object")
    end
    check(occursin(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", required(source, "repository", "source")), "Invalid repository")
    for key in ("commit_sha", "benchmark_source_sha")
        value = required(source, key, "source")
        check(value isa AbstractString && occursin(r"^[0-9a-fA-F]{40}$", value), "Invalid source.$key")
    end
    check(istext(required(source, "package_version", "source")), "Invalid package version")
    tag = required(source, "tag", "source")
    tag === nothing || version_tag(tag)
    check(required(source, "prerelease", "source") isa Bool, "source.prerelease must be boolean")
    check(!haskey(source, "working_tree_dirty") || source["working_tree_dirty"] isa Bool, "source.working_tree_dirty must be boolean")
    timestamp = required(run, "measured_at_utc", "run")
    check(timestamp isa AbstractString && occursin(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,3})?Z$", timestamp), "Measurement time must be ISO 8601 UTC")
    try
        DateTime(chop(timestamp; tail=1))
    catch
        error("Invalid UTC measurement time")
    end
    check(istext(required(run, "event_name", "run")), "Invalid event name")
    for key in ("workflow", "run_id")
        value = required(run, key, "run")
        check(optional(value, x -> istext(x) || iscount(x; positive=true)), "Invalid run.$key")
    end
    for key in ("run_number", "run_attempt")
        check(optional(required(run, key, "run"), x -> iscount(x; positive=true)), "Invalid run.$key")
    end
    url = required(run, "run_url", "run")
    check(optional(url, x -> x isa AbstractString && occursin(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/actions/runs/[0-9]+(?:/attempts/[0-9]+)?$", x)), "Invalid Actions URL")
    for name in ("cpu", "system", "runner", "runtime", "package_versions")
        check(isobject(required(environment, name, "environment")), "environment.$name must be an object")
    end
    cpu = environment["cpu"]
    models = required(cpu, "cpu_models", "cpu")
    check(models isa AbstractVector && all(istext, models), "Invalid CPU models")
    check(istext(required(cpu, "architecture", "cpu")), "Invalid CPU architecture")
    check(iscount(required(cpu, "logical_cpus_visible", "cpu"); positive=true), "Invalid visible logical CPU count")
    for key in ("physical_cores_visible", "affinity_cpu_count")
        check(optional(required(cpu, key, "cpu"), x -> iscount(x; positive=true)), "Invalid cpu.$key")
    end
    check(optional(required(cpu, "affinity_cpu_list", "cpu"), istext), "Invalid CPU affinity list")
    system = environment["system"]
    check(istext(required(system, "os", "system")), "Invalid system.os")
    check(optional(required(system, "kernel", "system"), istext), "Invalid system.kernel")
    check(optional(required(system, "memory_total_bytes_visible", "system"), x -> iscount(x; positive=true)), "Invalid visible memory")
    for key in ("label", "environment", "image_version")
        check(optional(required(environment["runner"], key, "runner"), istext), "Invalid runner.$key")
    end
    runtime = environment["runtime"]
    for key in ("julia_version", "blas_configuration")
        check(istext(required(runtime, key, "runtime")), "Invalid runtime.$key")
    end
    for key in ("julia_threads_default", "blas_threads")
        check(iscount(required(runtime, key, "runtime"); positive=true), "Invalid runtime.$key")
    end
    check(iscount(required(runtime, "julia_threads_interactive", "runtime")), "Invalid interactive thread count")
    check(optional(required(runtime, "julia_gc_threads", "runtime"), iscount), "Invalid GC thread count")
    check(all(value -> optional(value, istext), values(environment["package_versions"])), "Package versions must be strings or null")
    cases = required(report, "cases", "report")
    check(cases isa AbstractVector && !isempty(cases), "A measured report must contain cases")
    identifiers = Set{String}()
    for case in cases
        check(isobject(case), "Case must be an object")
        id = required(case, "case_id", "case")
        check(id isa AbstractString && occursin(r"^[A-Za-z0-9][A-Za-z0-9_.:/=+-]*$", id), "Invalid case ID")
        check(!(id in identifiers), "Duplicate case ID: $id")
        push!(identifiers, id)
        check(isobject(required(case, "parameters", id)), "Case parameters must be an object")
        check(!haskey(case, "description") || case["description"] isa AbstractString, "Invalid case description")
        measurement = required(case, "measurement_parameters", id)
        check(isobject(measurement), "Measurement parameters must be an object")
        check(iscount(required(measurement, "seed", id)), "Invalid seed")
        for key in ("evals", "samples_budget")
            check(iscount(required(measurement, key, id); positive=true), "Invalid $id.$key")
        end
        check(measurement["samples_budget"] >= 2, "Normal trials require a samples budget of at least two")
        budget = required(measurement, "seconds_budget", id)
        check(isnumber(budget) && budget > 0, "Invalid seconds budget")
        samples = required(case, "samples", id)
        check(iscount(samples; positive=true) && 2 <= samples <= measurement["samples_budget"], "Normal trials require at least two actual samples within the budget")
        check(isnumber(required(case, "median_time_ns", id)), "Invalid median time")
        for key in ("allocated_bytes", "allocations")
            check(iscount(required(case, key, id)), "Invalid $id.$key")
        end
    end
    return report
end

json_text(value) = JSON3.write(value) * "\n"
read_report(path) = validate_report(JSON3.read(read(path, String), Dict{String, Any}))
function write_json(path, value)
    mkpath(dirname(path))
    write(path, json_text(value))
    return path
end

html_escape(value) = replace(string(value), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;", '\'' => "&#39;")
function display_value(value)
    value === nothing && return "Not recorded"
    value isa Bool && return value ? "Yes" : "No"
    value isa AbstractDict && return "Structured parameter; see the raw data"
    value isa AbstractVector && return isempty(value) ? "Not recorded" : join(display_value.(value), ", ")
    return string(value)
end
markdown_escape(value) = replace(html_escape(display_value(value)), '|' => "\\|", '\n' => " ", '\r' => " ")

function time_value(value)
    value === nothing && return "Not recorded"
    divisor, unit = value >= 1e9 ? (1e9, "seconds") : value >= 1e6 ? (1e6, "milliseconds") : value >= 1e3 ? (1e3, "microseconds") : (1.0, "nanoseconds")
    number = round(value / divisor; sigdigits=5)
    text = isinteger(number) && abs(number) < typemax(Int) ? string(round(Int, number)) : string(number)
    return "$text $unit"
end

case_name(case) = workload_title(case)
ordered_cases(report) = sort(report["cases"]; by=case -> case["case_id"])

const STYLE = """
body{font:16px/1.55 system-ui,sans-serif;color:#162235;background:#f7f9fc;margin:0}main{max-width:1150px;margin:auto;padding:32px 24px}h1,h2{line-height:1.2}a{color:#1757a4}table{border-collapse:collapse;width:100%;background:white;margin:16px 0;display:block;overflow:auto}th,td{padding:10px 14px;text-align:left;border-bottom:1px solid #dce3eb;vertical-align:top}th{background:#eaf0f8}code,pre{font:13px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#edf2f8;padding:16px}dl{display:grid;grid-template-columns:minmax(140px,240px) 1fr;gap:8px 20px;background:white;padding:20px;border:1px solid #dce3eb}dt{font-weight:600}dd{margin:0;overflow-wrap:anywhere}.muted{color:#56657a}.notice{padding:12px 16px;background:#eaf0f8;border-left:4px solid #3576b6}nav{display:flex;gap:20px;flex-wrap:wrap}select{font:inherit;max-width:100%;padding:6px;margin:8px}svg{width:100%;height:auto;background:white;border:1px solid #dce3eb}details{margin:16px 0}summary{cursor:pointer} @media(max-width:600px){main{padding:20px 12px}dl{grid-template-columns:1fr;gap:2px}dd{margin-bottom:12px}}
"""

page(title, body; script="", lang="en") = "<!doctype html>\n<html lang=\"$(html_escape(lang))\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>$(html_escape(title))</title><style>$STYLE</style></head><body><main>$body</main>$script</body></html>\n"

function metadata_rows(report)
    source, run, environment = report["source"], report["run"], report["environment"]
    cpu, system, runner, runtime = (environment[key] for key in ("cpu", "system", "runner", "runtime"))
    rows = [
        "Repository" => source["repository"], "Source commit" => source["commit_sha"],
        "Benchmark definition commit" => source["benchmark_source_sha"],
        "Uncommitted changes" => get(source, "working_tree_dirty", nothing), "Version tag" => source["tag"],
        "Release type" => source["tag"] === nothing ? "Development or local build" :
            source["prerelease"] || !isempty(version_tag(source["tag"]).prerelease) ? "Prerelease" : "Stable release",
        "Algorithm library version" => source["package_version"], "Measured at (UTC)" => run["measured_at_utc"],
        "Processor model" => cpu["cpu_models"],
        "Processor architecture" => get(Dict("aarch64" => "64-bit ARM", "x86_64" => "64-bit x86"), cpu["architecture"], cpu["architecture"]),
        "Visible logical processors" => cpu["logical_cpus_visible"],
        "Processors available to this process" => cpu["affinity_cpu_count"], "Allowed processor identifiers" => cpu["affinity_cpu_list"],
        "Visible physical cores" => cpu["physical_cores_visible"],
        "Julia version" => runtime["julia_version"], "Julia computation threads" => runtime["julia_threads_default"],
        "Julia interactive threads" => runtime["julia_threads_interactive"], "Garbage collection threads" => runtime["julia_gc_threads"],
        "Matrix computation threads" => runtime["blas_threads"], "Matrix computation backend" => backend_name(runtime["blas_configuration"]),
        "Runner label" => runner["label"], "Runner type" => get(Dict("local"=>"Local runner", "github-hosted"=>"GitHub-hosted runner", "self-hosted"=>"Self-hosted runner"), runner["environment"], runner["environment"]),
        "Runner image version" => runner["image_version"], "Operating system" => get(Dict("Darwin"=>"macOS"),system["os"],system["os"]), "System kernel" => system["kernel"],
        "Visible system memory (bytes)" => system["memory_total_bytes_visible"], "Run trigger" => get(Dict("local"=>"Local run", "push"=>"Commit push", "release"=>"Release publication", "workflow_dispatch"=>"Manual run"),run["event_name"],run["event_name"]),
        "Automation workflow" => run["workflow"], "Run identifier" => run["run_id"], "Workflow run number" => run["run_number"],
        "Run attempt" => run["run_attempt"]]
    haskey(source,"suite_path") && push!(rows,"Benchmark definition file"=>source["suite_path"])
    haskey(source,"suite_origin") && push!(rows,"Benchmark definition source"=>get(Dict("checkout"=>"Current source checkout", "external"=>"External custom benchmark suite"),source["suite_origin"],source["suite_origin"]))
    return rows
end

function backend_name(configuration)
    for (pattern, name) in (("openblas", "OpenBLAS"), ("mkl", "Intel MKL"), ("accelerate", "Apple Accelerate"), ("blis", "BLIS"))
        occursin(pattern, lowercase(configuration)) && return name
    end
    return configuration
end

function package_rows(report)
    names = Dict("FiniteMPS"=>"Matrix product state library (FiniteMPS)",
                 "TensorKit"=>"Tensor computation library (TensorKit)",
                 "TensorOperations"=>"Tensor contraction library (TensorOperations)",
                 "MatrixAlgebraKit"=>"Matrix factorization library (MatrixAlgebraKit)",
                 "BenchmarkTools"=>"Timing tools (BenchmarkTools)")
    return [get(names,name,name)=>version for (name,version) in sort!(collect(report["environment"]["package_versions"]);by=first)]
end

function measurement_rows(case)
    parameters = case["measurement_parameters"]
    rows = Pair{String,Any}["Collected samples"=>case["samples"], "Evaluations per sample"=>parameters["evals"],
        "Warmup samples"=>get(parameters,"warmup_samples",nothing), "Random seed"=>parameters["seed"],
        "Sample limit"=>parameters["samples_budget"], "Time budget per case"=>"$(parameters["seconds_budget"]) seconds"]
    if get(parameters,"warmup_policy",nothing) == "smallest_size_per_group"
        push!(rows,"Compilation warmup"=>(parameters["warmup_samples"]==1 ?
            "One sample at the smallest input for this operation" :
            "Reused from the smallest input for this operation in the same Julia process"))
    end
    for (key,label) in (("mutates","Operation mutates its input"), ("gctrial","Garbage collection before each trial"), ("gc_after_case","Garbage collection after each completed case"), ("gcsample","Garbage collection before each sample"))
        haskey(parameters,key) && push!(rows,label=>parameters[key])
    end
    haskey(parameters,"overhead_ns") && push!(rows,"Timing overhead correction"=>time_value(parameters["overhead_ns"]))
    execution = get(case["parameters"],"execution",Dict())
    if execution isa AbstractDict
        for (key, value) in sort!(collect(execution); by=first)
            push!(rows, string(key) => value)
        end
    end
    return rows
end

function report_links(report; markdown=false)
    source, run = report["source"], report["run"]
    links = ["View measured source" => "https://github.com/$(source["repository"])/commit/$(source["commit_sha"])", "Download raw data (JSON)" => "report.json"]
    run["run_url"] === nothing || push!(links, "View workflow run" => run["run_url"])
    markdown && return join(["[$label]($url)" for (label, url) in links], " · ")
    return "<nav>" * join(["<a href=\"$(html_escape(url))\">$label</a>" for (label, url) in links]) * "</nav>"
end

function render_markdown(report)
    validate_report(report)
    io = IOBuffer()
    println(io, "# Performance measurement report\n\n", report_links(report; markdown=true))
    println(io, "\n## Measurement environment and thread settings\n\n| Field | Recorded at measurement time |\n| --- | --- |")
    for (key, value) in metadata_rows(report)
        println(io, "| ", key, " | ", markdown_escape(value), " |")
    end
    println(io, "\nProcessor counts and thread settings describe available resources, not runtime core utilization. System memory is the visible capacity; allocated memory is the amount allocated by the measured operation. Neither value is peak process memory.")
    println(io, "\n## Measurements\n\n| Operation and size | Median time | Total allocated bytes | Memory allocation count | Samples |\n| --- | ---: | ---: | ---: | ---: |")
    for case in ordered_cases(report)
        values = (case_name(case),time_value(case["median_time_ns"]),case["allocated_bytes"],case["allocations"],case["samples"])
        println(io, "| ", join(markdown_escape.(values), " | "), " |")
    end
    for case in ordered_cases(report)
        println(io, "\n### ", markdown_escape(case_name(case)), "\n\n", markdown_escape(workload_description(case)), "\n")
        for (heading,rows) in (("Workload details",workload_facts(case)),("Sampling and execution settings",measurement_rows(case)))
            isempty(rows) && continue
            println(io, "\n#### ",heading,"\n\n| Field | Value |\n| --- | --- |")
            for (label,value) in rows
                println(io,"| ",markdown_escape(label)," | ",markdown_escape(value)," |")
            end
        end
    end
    println(io, "\n## Software versions\n\n| Software | Version |\n| --- | --- |")
    for (name,version) in package_rows(report)
        println(io,"| ",markdown_escape(name)," | ",markdown_escape(version)," |")
    end
    println(io,"\nComplete case identifiers and original parameters are available in the [raw data file](report.json).")
    return String(take!(io))
end

function render_report(report)
    validate_report(report)
    io = IOBuffer()
    identity = something(report["source"]["tag"], "Development or local build")
    println(io, """<style>.resource-summary{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 28px;margin:12px 0}.resource-summary dt{font-size:13px;color:#56657a}.resource-summary dd{margin:0}.report-identity{overflow-wrap:anywhere;margin:12px 0}.report-identity code{font-size:14px}@media(max-width:600px){.resource-summary{grid-template-columns:1fr;gap:10px}}</style>""")
    println(io, "<h1>Performance measurement report: ", html_escape(identity), "</h1>", report_links(report))
    rows = metadata_rows(report)
    values = Dict(rows)
    shown = Set(["Source commit", "Measured at (UTC)", "Release type"])
    println(io, "<p class=\"report-identity\"><strong>Source commit:</strong> <code>", html_escape(values["Source commit"]),
            "</code><br><strong>Measured at (UTC):</strong> ", html_escape(values["Measured at (UTC)"]),
            " · ", html_escape(values["Release type"]), "</p>")
    if get(report["source"], "working_tree_dirty", false)
        println(io, "<p class=\"notice\"><strong>Uncommitted changes:</strong> This measurement used uncommitted source code or benchmark definitions.</p>")
        push!(shown, "Uncommitted changes")
    end
    println(io, "<h2>Measurement environment and thread settings</h2><dl class=\"resource-summary\">")
    essentials = ("Processor model", "Julia version", "Visible logical processors", "Processors available to this process",
                  "Julia computation threads", "Julia interactive threads", "Matrix computation threads", "Runner label", "Operating system")
    for key in essentials
        println(io, "<div><dt>", html_escape(key), "</dt><dd>", html_escape(display_value(values[key])), "</dd></div>")
        push!(shown, key)
    end
    println(io, "</dl><details><summary>Additional run metadata</summary><dl>")
    for (key, value) in rows
        key in shown && continue
        println(io, "<dt>", html_escape(key), "</dt><dd>", html_escape(display_value(value)), "</dd>")
    end
    println(io, "</dl></details><p class=\"muted\">Processor counts and thread settings describe available resources, not runtime core utilization. Visible system memory is the system capacity.</p>")
    println(io, "<h2>Measurements</h2><table><thead><tr><th>Operation and size</th><th>Median time</th><th>Total allocated bytes</th><th>Memory allocation count</th><th>Samples</th></tr></thead><tbody>")
    for case in ordered_cases(report)
        measurements = (case_name(case),time_value(case["median_time_ns"]),case["allocated_bytes"],case["allocations"],case["samples"])
        println(io, "<tr>", join(["<td>$(html_escape(value))</td>" for value in measurements]), "</tr>")
    end
    println(io, "</tbody></table><p>Allocated memory is the amount allocated by the measured operation, not peak process memory.</p>")
    for case in ordered_cases(report)
        println(io, "<details><summary>", html_escape(case_name(case)), ": Workload and measurement settings</summary><p>", html_escape(workload_description(case)), "</p>")
        for (heading,rows) in (("Workload details",workload_facts(case)),("Sampling and execution settings",measurement_rows(case)))
            isempty(rows) && continue
            println(io,"<h3>",heading,"</h3><dl>")
            for (label,value) in rows
                println(io,"<dt>",html_escape(label),"</dt><dd>",html_escape(display_value(value)),"</dd>")
            end
            println(io,"</dl>")
        end
        println(io,"</details>")
    end
    println(io, "<details><summary>Software versions</summary><dl>")
    for (name,version) in package_rows(report)
        println(io,"<dt>",html_escape(name),"</dt><dd>",html_escape(display_value(version)),"</dd>")
    end
    println(io,"</dl></details><p>Complete case identifiers and original parameters are available in the <a href=\"report.json\">raw data file</a>.</p>")
    return page("Performance measurement report: $identity", String(take!(io)))
end

end # module
