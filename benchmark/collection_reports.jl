module PerformanceCollections

include("reports.jl")
using .PerformanceReports
using JSON3

export read_collection, validate_collection, render_collection, render_collection_markdown,
       build_collection, configuration_id, collection_record, collection_files

configuration_id(threads::Integer) = "julia-$threads-blas-1"
const THREADS = (1, 2, 4)
check(condition, message) = condition || error(message)

function validate_index(index)
    check(index isa AbstractDict, "Collection must be a JSON object")
    check(get(index, "schema_version", nothing) === 1 && get(index, "kind", nothing) == "performance_collection",
          "Unsupported collection schema")
    check(get(index, "status", nothing) == "complete", "Collection is not complete")
    check(get(index, "source", nothing) isa AbstractDict && get(index, "run", nothing) isa AbstractDict,
          "Collection must preserve source and run identity")
    configurations = get(index, "configurations", nothing)
    check(configurations isa AbstractVector && length(configurations) == 3, "Collection requires exactly Julia 1, 2, and 4 configurations")
    for (configuration, threads) in zip(configurations, THREADS)
        check(configuration isa AbstractDict, "Configuration must be an object")
        check(get(configuration, "julia_threads", nothing) === threads, "Configurations must be ordered Julia 1, 2, 4")
        check(get(configuration, "blas_threads", nothing) === 1 && get(configuration, "gc_threads", nothing) === 1,
              "Collection requires BLAS 1 and GC 1")
        check(get(configuration, "report_path", nothing) == "configurations/$(configuration_id(threads))/report.json",
              "Configuration report path must match its thread configuration")
    end
    return index
end

without(object, keys) = Dict(key => value for (key, value) in object if key ∉ keys)
function case_definition(case)
    result = without(case, ("samples", "median_time_ns", "allocated_bytes", "allocations"))
    result["parameters"] = without(case["parameters"], ("execution",))
    return result
end

function validate_collection(index, reports::AbstractVector)
    validate_index(index)
    check(length(reports) == 3, "Collection must have three measured reports")
    foreach(validate_report, reports)
    first_report = first(reports)
    check(index["source"] == first_report["source"] && index["run"] == first_report["run"],
          "Collection identity must equal the first measuring process report")
    reference_cases = Dict(case["case_id"] => case_definition(case) for case in first_report["cases"])
    for (report, threads) in zip(reports, THREADS)
        check(report["source"] == index["source"], "Collection source identity differs between configurations")
        check(without(report["run"], ("measured_at_utc",)) == without(index["run"], ("measured_at_utc",)),
              "Collection run identity differs between configurations")
        environment = report["environment"]
        reference_environment = first_report["environment"]
        check(without(environment, ("runtime",)) == without(reference_environment, ("runtime",)),
              "Collection machine/package metadata differs between configurations")
        runtime = environment["runtime"]
        check(runtime["julia_threads_default"] == threads && runtime["julia_threads_interactive"] == 0 &&
              runtime["julia_gc_threads"] == 1 && runtime["blas_threads"] == 1,
              "Actual Julia/interactive/GC/BLAS configuration differs from declaration")
        check(without(runtime, ("julia_threads_default",)) == without(reference_environment["runtime"], ("julia_threads_default",)),
              "Undeclared runtime configuration differs between reports")
        actual_cases = Dict(case["case_id"] => case_definition(case) for case in report["cases"])
        check(actual_cases == reference_cases, "Collection case IDs, definitions, dimensions, seeds, or measurement parameters differ")
    end
    return (index=index, reports=reports)
end

function read_collection(path::AbstractString)
    islink(path) && error("Collection index cannot be a symbolic link")
    index = validate_index(JSON3.read(read(path, String), Dict{String, Any}))
    root = realpath(dirname(abspath(path)))
    reports = Any[]
    for configuration in index["configurations"]
        relative = configuration["report_path"]
        current = root
        for part in split(relative, '/')
            current = joinpath(current, part)
            islink(current) && error("Configuration reports cannot follow symbolic links")
        end
        push!(reports, read_report(current))
    end
    validate_collection(index, reports)
    return (index=index, reports=reports, root=root)
end

function collection_record(bundle)
    validate_collection(bundle.index, bundle.reports)
    # Internal history view only: each thread/case pair remains its own series.
    cases = Any[]
    for (report, threads) in zip(bundle.reports, THREADS), case in report["cases"]
        item = deepcopy(case)
        item["case_id"] = "$(configuration_id(threads))/$(case["case_id"])"
        push!(cases, item)
    end
    return Dict("source" => bundle.index["source"], "run" => bundle.index["run"], "cases" => cases,
                "kind" => "performance_collection")
end

time_text(value) = time_value(value)

const METRIC_SCRIPT = raw"""
<script>
(() => {
  const selector = document.getElementById('metric');
  function show() {
    document.querySelectorAll('td[data-time]').forEach(cell => {
      const time = Number(cell.dataset.time), reference = Number(cell.dataset.reference);
      if (selector.value === 'speedup') cell.textContent = time > 0 ? (reference / time).toPrecision(4) + ' ×' : 'Unavailable (measured time is zero)';
      else if (selector.value === 'bytes') cell.textContent = cell.dataset.bytes + ' bytes';
      else if (selector.value === 'allocations') cell.textContent = cell.dataset.allocations + ' allocations';
      else {
        const divisor = time >= 1e9 ? 1e9 : time >= 1e6 ? 1e6 : time >= 1e3 ? 1e3 : 1;
        const unit = divisor === 1e9 ? 'seconds' : divisor === 1e6 ? 'milliseconds' : divisor === 1e3 ? 'microseconds' : 'nanoseconds';
        cell.textContent = Number((time / divisor).toPrecision(5)).toString() + ' ' + unit;
      }
    });
  }
  selector.addEventListener('change', show);
})();
</script>
"""

function render_facts(io, rows)
    println(io, "<dl>")
    for (label, value) in rows
        println(io, "<dt>", html_escape(label), "</dt><dd>", html_escape(display_value(value)), "</dd>")
    end
    println(io, "</dl>")
end

function render_collection(bundle)
    validate_collection(bundle.index, bundle.reports)
    first_report = first(bundle.reports)
    source, run = bundle.index["source"], bundle.index["run"]
    io = IOBuffer()
    println(io, "<h1>Performance report</h1><p><strong>", html_escape(something(source["tag"], "Development or local build")),
        "</strong></p>")
    println(io, "<p>This report contains ", sum(length(report["cases"]) for report in bundle.reports), " measurements.</p>")
    println(io, "<p>First measurement timestamp (UTC): ", html_escape(run["measured_at_utc"]), ". Julia uses 1, 2, or 4 compute threads; the linear algebra backend and garbage collector each use 1 thread.</p>")
    get(source, "working_tree_dirty", false) && println(io, "<p class=\"notice\">These measurements include uncommitted changes in the source checkout.</p>")
    println(io, "<nav><a href=\"collection.json\">Download the raw data index</a>")
    for threads in THREADS
        id = configuration_id(threads)
        println(io, "<a href=\"configurations/$id/index.html\">Detailed report: $threads ", threads == 1 ? "thread" : "threads", "</a>")
    end
    println(io, "</nav><p><label for=\"metric\">Display metric: </label><select id=\"metric\"><option value=\"time\">Median execution time</option><option value=\"speedup\">Speedup relative to one thread</option><option value=\"bytes\">Total allocated bytes</option><option value=\"allocations\">Memory allocation count</option></select></p>")
    println(io, "<p class=\"muted\">Each row is one benchmark case and each column is a thread configuration. Speedup compares the same case against its single-thread measurement. Allocated bytes measure cumulative allocation, not peak memory.</p>")
    cpu = first_report["environment"]["cpu"]
    println(io, "<p>Processor: ", html_escape(join(cpu["cpu_models"], ", ")),
        "; visible logical processors: ", cpu["logical_cpus_visible"], ".</p>")
    println(io, "<details><summary>Source version, hardware, and runtime settings</summary>")
    for (report, threads) in zip(bundle.reports, THREADS)
        println(io, "<h3>Runtime with $threads ", threads == 1 ? "thread" : "threads", "</h3>")
        render_facts(io, metadata_rows(report))
    end
    println(io, "<h3>Software versions</h3>")
    render_facts(io, package_rows(first_report))
    println(io, "</details>")
    lookups = [Dict(case["case_id"] => case for case in report["cases"]) for report in bundle.reports]
    println(io, "<table><thead><tr><th>Benchmark case</th><th>1 thread</th><th>2 threads</th><th>4 threads</th></tr></thead><tbody>")
    for case in ordered_cases(first_report)
        println(io, "<tr><td>", html_escape(case_name(case)), "</td>")
        for lookup in lookups
            value = lookup[case["case_id"]]
            println(io, "<td data-time=\"", value["median_time_ns"], "\" data-reference=\"", case["median_time_ns"],
                "\" data-bytes=\"", value["allocated_bytes"], "\" data-allocations=\"", value["allocations"], "\">", time_text(value["median_time_ns"]), "</td>")
        end
        println(io, "</tr>")
    end
    println(io, "</tbody></table>")
    for case in ordered_cases(first_report)
        println(io, "<details><summary>", html_escape(case_name(case)), ": Input and sampling settings</summary><p>", html_escape(workload_description(case)), "</p>")
        render_facts(io, workload_facts(case))
        for (lookup, threads) in zip(lookups, THREADS)
            println(io, "<h3>Sampling with $threads ", threads == 1 ? "thread" : "threads", "</h3>")
            render_facts(io, measurement_rows(lookup[case["case_id"]]))
        end
        println(io, "</details>")
    end
    return page("Performance report", String(take!(io)); script=METRIC_SCRIPT)
end

function render_collection_markdown(bundle)
    validate_collection(bundle.index, bundle.reports)
    io = IOBuffer()
    println(io, "# Performance report\n\nSource commit: `", bundle.index["source"]["commit_sha"], "`\n\nJulia uses 1, 2, or 4 compute threads; the linear algebra backend and garbage collector each use 1 thread.\n")
    println(io, "| Thread configuration | Measurements | Measurement timestamp (UTC) | Detailed report |\n| --- | ---: | --- | --- |")
    for (report, threads) in zip(bundle.reports, THREADS)
        println(io, "| $threads ", threads == 1 ? "thread" : "threads", " | ", length(report["cases"]), " | ", report["run"]["measured_at_utc"], " | [Input and sampling details](configurations/$(configuration_id(threads))/report.md) |")
    end
    lookups = [Dict(case["case_id"]=>case for case in report["cases"]) for report in bundle.reports]
    println(io, "\n| Benchmark case | 1 thread | 2 threads | 4 threads |\n| --- | ---: | ---: | ---: |")
    for case in ordered_cases(first(bundle.reports))
        values = [time_text(lookup[case["case_id"]]["median_time_ns"]) for lookup in lookups]
        println(io, "| ", PerformanceReports.markdown_escape(case_name(case)), " | ", join(values, " | "), " |")
    end
    return String(take!(io))
end

function collection_files(bundle)
    validate_collection(bundle.index, bundle.reports)
    files = Dict("collection.json" => json_text(bundle.index), "index.html" => render_collection(bundle),
                 "report.md" => render_collection_markdown(bundle))
    for (configuration, report) in zip(bundle.index["configurations"], bundle.reports)
        path = configuration["report_path"]
        files[path] = json_text(report)
        files[joinpath(dirname(path), "index.html")] = render_report(report)
        files[joinpath(dirname(path), "report.md")] = render_markdown(report)
    end
    return files
end

function build_collection(path; output=dirname(abspath(path)))
    bundle = read_collection(path)
    files = collection_files(bundle)
    return write_build_files(files, output)
end

function write_build_files(files, output)
    # Preflight all owned output paths before writing. The input directory can be
    # reused, or JSON and self-contained HTML can be copied to a fresh output.
    output = abspath(output)
    islink(output) && error("HTML output root cannot be a symbolic link")
    isdir(output) && (output = realpath(output))
    for relative in keys(files)
        destination = abspath(joinpath(output, relative))
        current = destination
        while current != dirname(output)
            islink(current) && error("HTML output cannot follow symbolic links")
            ispath(current) && (current == destination ? !isfile(current) : !isdir(current)) && error("HTML output path collision")
            parent = dirname(current)
            parent == current && break
            current = parent
        end
    end
    for (relative, content) in files
        destination = joinpath(output, relative)
        mkpath(dirname(destination))
        write(destination, content)
    end
    return joinpath(abspath(output), "index.html")
end

end # module
