module PerformancePublisher

include("reports.jl")
using .PerformanceReports
using JSON3
include("collection_reports.jl")
using .PerformanceCollections

export publish_report, publish_collection, trend_data

function assert_safe_path(path)
    current = abspath(path)
    while true
        islink(current) && error("Refusing symbolic link in publication path: $current")
        parent = dirname(current)
        parent == current && break
        current = parent
    end
end

function inspect_site(site)
    isdir(site) || error("Site checkout must be an existing directory")
    # /tmp may itself be a platform alias. Resolve the supplied checkout root, then
    # reject symlinks for every path this publisher owns inside that checkout.
    islink(site) && error("Site checkout cannot be a symbolic link")
    root = joinpath(realpath(site), "performance")
    assert_safe_path(root)
    ispath(root) && !isdir(root) && error("performance must be a directory")
    if isdir(root)
        for (directory, directories, files) in walkdir(root; follow_symlinks=false)
            for name in vcat(directories, files)
                islink(joinpath(directory, name)) && error("Symbolic links are not supported inside performance")
            end
        end
    end
    return root
end

semver_key(tag) = VersionNumber(first(split(tag[2:end], '+')))
is_prerelease(report) = report["source"]["prerelease"] || !isempty(version_tag(report["source"]["tag"]).prerelease)

function read_snapshot(directory)
    isdir(directory) && isfile(joinpath(directory, "index.html")) || error("Incomplete performance snapshot")
    single, collection = joinpath(directory, "report.json"), joinpath(directory, "collection.json")
    xor(isfile(single), isfile(collection)) || error("Snapshot must contain exactly one report or collection")
    if isfile(single)
        report = read_report(single)
        return (record=report, payload=report)
    end
    bundle = read_collection(collection)
    for config in bundle.index["configurations"]
        isfile(joinpath(directory, dirname(config["report_path"]), "index.html")) || error("Incomplete configuration HTML")
    end
    return (record=PerformanceCollections.collection_record(bundle),
            payload=Dict("index" => bundle.index, "reports" => bundle.reports))
end

function read_archives(root)
    releases = joinpath(root, "releases")
    reports = Dict{String, Any}()
    if !ispath(releases)
        return reports
    end
    if !isdir(releases)
        @warn "Performance releases path is not a directory; ignoring historical archives: $releases"
        return reports
    end
    for tag in readdir(releases)
        directory = joinpath(releases, tag)
        if !isdir(directory)
            @warn "Skipping performance release entry that is not a directory: $tag"
            continue
        end
        report = try
            parsed = read_snapshot(directory).record
            version_tag(tag)
            parsed["source"]["tag"] == tag || error("Archive tag and directory disagree: $tag")
            parsed
        catch exception
            @warn "Skipping unreadable performance release archive: $tag" exception
            continue
        end
        reports[tag] = report
    end
    return reports
end

sorted_tags(archives) = sort!(collect(keys(archives)); by=tag -> (semver_key(tag), tag))

function trend_case_name(case, record)
    execution = get(get(case, "parameters", Dict()), "execution", Dict())
    execution isa AbstractDict || (execution = Dict())
    runtime = get(get(record, "environment", Dict()), "runtime", Dict())
    julia_threads = get(execution, "julia_threads", get(runtime, "julia_threads_default", nothing))
    blas_threads = get(execution, "blas_threads", get(runtime, "blas_threads", nothing))
    # Collection association keys preserve each measuring process configuration.
    prefix = match(r"^julia-([0-9]+)-blas-([0-9]+)/", case["case_id"])
    if prefix !== nothing
        julia_threads === nothing && (julia_threads = prefix.captures[1])
        blas_threads === nothing && (blas_threads = prefix.captures[2])
    end
    threads = julia_threads === nothing ? "Julia thread count not recorded" : "Julia threads: $julia_threads"
    blas_threads === nothing || (threads *= ", BLAS threads: $blas_threads")
    return case_name(case) * " · " * threads
end

function trend_data(archives)
    tags = sorted_tags(archives)
    ids = sort!(unique([case["case_id"] for tag in tags for case in archives[tag]["cases"]]))
    releases = [Dict("tag" => tag, "commit_sha" => archives[tag]["source"]["commit_sha"],
                     "measured_at_utc" => archives[tag]["run"]["measured_at_utc"],
                     "prerelease" => is_prerelease(archives[tag]),
                     "url" => "../releases/$tag/") for tag in tags]
    series = Any[]
    for id in ids
        points = Any[]
        display_name = ""
        for tag in tags
            index = findfirst(case -> case["case_id"] == id, archives[tag]["cases"])
            if index === nothing
                push!(points, nothing)
            else
                case = archives[tag]["cases"][index]
                display_name = trend_case_name(case, archives[tag])
                push!(points, Dict(key => case[key] for key in ("median_time_ns", "allocated_bytes", "allocations")))
            end
        end
        push!(series, Dict("case_id" => id, "display_name" => display_name, "points" => points))
    end
    return Dict("schema_version" => 1, "releases" => releases, "cases" => series)
end

const METRICS = ["median_time_ns" => "Median time (nanoseconds)", "allocated_bytes" => "Total allocated bytes", "allocations" => "Memory allocation count"]

function initial_graph(data)
    isempty(data["cases"]) && return "<p>No successful release measurements have been archived yet.</p>"
    series, releases = first(data["cases"]), data["releases"]
    points = series["points"]
    values = [point["median_time_ns"] for point in points if point !== nothing]
    ceiling = max(maximum(values), 1)
    xs(i) = length(points) == 1 ? 460.0 : 75.0 + (i - 1) * 770.0 / (length(points) - 1)
    ys(value) = 265.0 - value / ceiling * 220.0
    io = IOBuffer()
    println(io, "<svg id=\"plot\" viewBox=\"0 0 920 340\" role=\"img\" aria-label=\"Performance history\"><title>", html_escape(series["display_name"]), ": Median time (nanoseconds)</title><line x1=\"75\" y1=\"265\" x2=\"845\" y2=\"265\" stroke=\"#72829a\"/><text x=\"10\" y=\"45\">", ceiling, "</text><text x=\"35\" y=\"270\">0</text>")
    previous = nothing
    for (i, point) in enumerate(points)
        if point === nothing
            previous = nothing
        else
            x, y = xs(i), ys(point["median_time_ns"])
            previous === nothing || println(io, "<line x1=\"", previous[1], "\" y1=\"", previous[2], "\" x2=\"$x\" y2=\"$y\" stroke=\"#1757a4\" stroke-width=\"2\"/>")
            println(io, "<a href=\"", html_escape(releases[i]["url"]), "\"><circle cx=\"$x\" cy=\"$y\" r=\"5\" fill=\"#1757a4\"><title>", html_escape(releases[i]["tag"]), ": ", point["median_time_ns"], " nanoseconds</title></circle></a>")
            previous = (x, y)
        end
        println(io, "<text x=\"", xs(i), "\" y=\"295\" text-anchor=\"middle\" font-size=\"12\">", html_escape(releases[i]["tag"]), "</text>")
    end
    println(io, "</svg>")
    return String(take!(io))
end

function initial_table(data)
    isempty(data["cases"]) && return ""
    io = IOBuffer()
    println(io, "<table id=\"values\"><thead><tr><th>Version</th><th>Release type</th><th>Median time (nanoseconds)</th><th>Total allocated bytes</th><th>Memory allocation count</th></tr></thead><tbody>")
    for (release, point) in zip(data["releases"], first(data["cases"])["points"])
        println(io, "<tr><td><a href=\"", html_escape(release["url"]), "\">", html_escape(release["tag"]), "</a></td><td>", release["prerelease"] ? "Prerelease" : "Stable release", "</td>")
        for (metric, _) in METRICS
            println(io, "<td>", point === nothing ? "No measurement" : string(point[metric]), "</td>")
        end
        println(io, "</tr>")
    end
    println(io, "</tbody></table>")
    return String(take!(io))
end

const TREND_SCRIPT = raw"""
<script>
(() => {
  const data = JSON.parse(document.getElementById('trend-data').textContent);
  if (!data.cases.length) return;
  const caseSelect = document.getElementById('case'), metricSelect = document.getElementById('metric');
  const svg = document.getElementById('plot'), tbody = document.querySelector('#values tbody');
  const ns = 'http://www.w3.org/2000/svg';
  function node(name, attributes, text) {
    const element = document.createElementNS(ns, name);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function update() {
    const series = data.cases[Number(caseSelect.value)], metric = metricSelect.value;
    const values = series.points.filter(p => p !== null).map(p => p[metric]);
    const ceiling = Math.max(1, ...values);
    const x = i => series.points.length === 1 ? 460 : 75 + i * 770 / (series.points.length - 1);
    const y = value => 265 - value / ceiling * 220;
    svg.replaceChildren(node('title', {}, `${series.display_name}: ${metricSelect.selectedOptions[0].textContent}`));
    svg.append(node('line', {x1:75,y1:265,x2:845,y2:265,stroke:'#72829a'}));
    svg.append(node('text', {x:10,y:45,'font-size':13}, ceiling.toPrecision(4)));
    svg.append(node('text', {x:35,y:270,'font-size':13}, '0'));
    let previous = null;
    series.points.forEach((point, i) => {
      const release = data.releases[i];
      if (point === null) previous = null;
      else {
        const current = [x(i), y(point[metric])];
        if (previous !== null) svg.append(node('line', {x1:previous[0],y1:previous[1],x2:current[0],y2:current[1],stroke:'#1757a4','stroke-width':2}));
        const link = node('a', {href:release.url});
        const circle = node('circle', {cx:current[0],cy:current[1],r:5,fill:'#1757a4'});
        circle.append(node('title', {}, `${release.tag}: ${metricSelect.selectedOptions[0].textContent} ${point[metric]}`));
        link.append(circle); svg.append(link); previous = current;
      }
      // Keep labels readable for long histories; every release remains in the table.
      const step = Math.max(1, Math.ceil(data.releases.length / 7));
      if (i % step === 0 || i === data.releases.length - 1)
        svg.append(node('text', {x:x(i),y:295,'text-anchor':'middle','font-size':12}, release.tag));
    });
    tbody.replaceChildren();
    data.releases.forEach((release, i) => {
      const row = document.createElement('tr'), linkCell = document.createElement('td'), link = document.createElement('a');
      link.href = release.url; link.textContent = release.tag; linkCell.append(link); row.append(linkCell);
      const point = series.points[i];
      const cells = [release.prerelease ? 'Prerelease' : 'Stable release', ...['median_time_ns','allocated_bytes','allocations'].map(key => point === null ? 'No measurement' : String(point[key]))];
      for (const value of cells) { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }
      tbody.append(row);
    });
  }
  caseSelect.addEventListener('change', update); metricSelect.addEventListener('change', update); update();
})();
</script>
"""

function render_trend(data)
    io = IOBuffer()
    println(io, "<h1>Performance history</h1><nav><a href=\"../\">Performance report home</a><a href=\"data.json\">Download historical data</a></nav><p class=\"notice\">Historical measurements are shown without adjustment for differences in the execution environment. Missing measurements appear as gaps; zero allocations are measured values. Each metric is presented separately, with no aggregate score.</p>")
    if !isempty(data["cases"])
        println(io, "<label for=\"case\">Workload and execution settings</label><select id=\"case\">")
        for (index, case) in enumerate(data["cases"])
            println(io, "<option value=\"", index - 1, "\">", html_escape(case["display_name"]), "</option>")
        end
        println(io, "</select><label for=\"metric\">Metric</label><select id=\"metric\">")
        for (key, label) in METRICS
            println(io, "<option value=\"$key\">$label</option>")
        end
        println(io, "</select>")
    end
    print(io, initial_graph(data), initial_table(data))
    # JSON in a script element must not contain a literal HTML closing tag.
    payload = replace(JSON3.write(data), '<' => "\\u003c", '>' => "\\u003e", '&' => "\\u0026", '\u2028' => "\\u2028", '\u2029' => "\\u2029")
    script = "<script id=\"trend-data\" type=\"application/json\">$payload</script>" * TREND_SCRIPT
    return page("Performance history", String(take!(io)); script)
end

function render_index(archives, branches)
    io = IOBuffer()
    println(io, "<h1>Performance reports</h1><nav><a href=\"../\">Project documentation</a><a href=\"stable/\">Latest stable release</a><a href=\"trend/\">Performance history</a></nav>")
    for branch in ("dev", "main")
        report = get(branches, branch, nothing)
        if report === nothing
            println(io, "<h2>$branch</h2><p>No measurements yet.</p>")
        else
            println(io, "<h2><a href=\"$branch/\">$branch</a></h2><p>Commit <code>", html_escape(report["source"]["commit_sha"]), "</code>, measured at ", html_escape(report["run"]["measured_at_utc"]), ".</p>")
        end
    end
    println(io, "<h2>Release archives</h2>")
    if isempty(archives)
        println(io, "<p>No successful release measurements have been archived yet.</p>")
    else
        println(io, "<table><thead><tr><th>Version</th><th>Release type</th><th>Measured at (UTC)</th><th>Commit</th></tr></thead><tbody>")
        for tag in reverse(sorted_tags(archives))
            report = archives[tag]
            println(io, "<tr><td><a href=\"releases/$tag/\">$tag</a></td><td>", is_prerelease(report) ? "Prerelease" : "Stable release", "</td><td>", html_escape(report["run"]["measured_at_utc"]), "</td><td><code>", html_escape(report["source"]["commit_sha"]), "</code></td></tr>")
        end
        println(io, "</tbody></table>")
    end
    return page("Performance reports", String(take!(io)))
end

function render_stable(archives)
    stable = filter(tag -> !is_prerelease(archives[tag]), sorted_tags(archives))
    isempty(stable) && return page("Latest stable release performance report", "<h1>Latest stable release performance report</h1><p>No stable release measurements have been archived yet.</p><a href=\"../\">Performance report home</a>")
    tag = last(stable)
    return page("Latest stable release performance report", "<h1>Latest stable release performance report</h1><p>The highest successfully archived stable release is <a href=\"../releases/$tag/\">$tag</a>.</p><a href=\"../\">Performance report home</a>")
end

"""Plan and validate the complete update before writing only site/performance.

Call while holding the shared site publishing lock, against the latest checkout.
No git operations, runtime metadata collection, or benchmark imports occur here.
"""
function publish_report(report_path, site; mode, dry_run=false, expected_sha=nothing, expected_tag=nothing)
    report = read_report(report_path)
    files = Dict("report.json" => json_text(report), "index.html" => render_report(report))
    return publish_snapshot(report, report, files, site; mode, dry_run, expected_sha, expected_tag)
end

function publish_collection(collection_path, site; mode, dry_run=false, expected_sha=nothing, expected_tag=nothing)
    bundle = read_collection(collection_path)
    record = PerformanceCollections.collection_record(bundle)
    payload = Dict("index" => bundle.index, "reports" => bundle.reports)
    return publish_snapshot(record, payload, collection_files(bundle), site; mode, dry_run, expected_sha, expected_tag)
end

function publish_snapshot(report, payload, files, site; mode, dry_run, expected_sha, expected_tag)
    mode in ("dev", "main", "release") || error("Mode must be dev, main or release")
    source, run = report["source"], report["run"]
    expected_sha === nothing || source["commit_sha"] == expected_sha || error("Report SHA does not match expected target")
    expected_tag === nothing || source["tag"] == expected_tag || error("Report tag does not match expected target")
    if mode == "release"
        source["tag"] === nothing && error("Release publication requires a version tag")
        version_tag(source["tag"])
    else
        source["tag"] === nothing || error("Branch publication cannot have a tag")
        run["event_name"] == "push" || error("Branch publication requires a push report")
        all(run[key] !== nothing for key in ("workflow", "run_id", "run_number", "run_attempt")) || error("Branch publication requires workflow ordering metadata")
    end
    root = inspect_site(site)
    archives = read_archives(root)
    branches = Dict{String, Any}()
    previous_snapshot = nothing
    for branch in ("dev", "main")
        path = joinpath(root, branch)
        ispath(path) || continue
        snapshot = try
            read_snapshot(path)
        catch exception
            @warn "Ignoring unreadable $branch snapshot." exception
            nothing
        end
        snapshot === nothing && continue
        branches[branch] = snapshot.record
        branch == mode && (previous_snapshot = snapshot)
    end
    for old in values(archives)
        old["source"]["repository"] == source["repository"] || error("Mixed repositories in release archives")
    end
    for (branch, record) in branches
        record["source"]["tag"] === nothing || error("$branch snapshot contains a tag")
        record["source"]["repository"] == source["repository"] || error("$branch belongs to another repository")
    end
    planned = Dict{String, String}()
    if mode == "release"
        tag = source["tag"]
        if haskey(archives, tag)
            archives[tag]["source"]["commit_sha"] == source["commit_sha"] || error("Tag $tag moved: existing archive has a different SHA")
            return "already-archived"
        end
        archives[tag] = report
        snapshot_path = "releases/$tag"
    else
        if previous_snapshot !== nothing
            previous = previous_snapshot.record["run"]
            previous["workflow"] == run["workflow"] || error("$mode workflow identity changed; run numbers are not comparable")
            all(previous[key] !== nothing for key in ("run_id", "run_number", "run_attempt")) || error("Existing $mode lacks ordering metadata")
            old_order, new_order = (previous["run_number"], previous["run_attempt"]), (run["run_number"], run["run_attempt"])
            old_order > new_order && return "older-$mode-skipped"
            if previous["run_number"] == run["run_number"]
                previous["run_id"] == run["run_id"] || error("Same run number has a different run ID")
                previous_snapshot.record["source"]["commit_sha"] == source["commit_sha"] || error("Rerun changed the measured SHA")
            end
            if old_order == new_order
                previous_snapshot.payload == payload && return "no-op"
                error("A completed run attempt already has a different $mode report")
            end
        end
        branches[mode] = report
        snapshot_path = mode
    end
    for (relative, contents) in files
        planned[joinpath(snapshot_path, relative)] = contents
    end
    planned["index.html"] = render_index(archives, branches)
    if mode == "release" || !isfile(joinpath(root, "trend", "data.json"))
        data = trend_data(archives)
        planned["trend/data.json"] = json_text(data)
        planned["trend/index.html"] = render_trend(data)
        planned["stable/index.html"] = render_stable(archives)
    end
    # Preflight all destinations, including parent file/directory collisions, before
    # the first write. Archive JSON and HTML remain immutable after this operation.
    for relative in keys(planned)
        destination = joinpath(root, relative)
        assert_safe_path(destination)
        ispath(destination) && !isfile(destination) && error("Destination is not a regular file: $relative")
        parent = dirname(destination)
        while parent != dirname(root)
            ispath(parent) && !isdir(parent) && error("Destination parent is not a directory")
            parent = dirname(parent)
        end
    end
    changed = filter(pair -> !isfile(joinpath(root, first(pair))) || read(joinpath(root, first(pair)), String) != last(pair), planned)
    isempty(changed) && return "no-op"
    dry_run && return "published"
    # Install a complete snapshot directory together, so changing a branch between the
    # single-report and collection formats cannot leave a stale second manifest.
    destination = joinpath(root, snapshot_path)
    mkpath(dirname(destination))
    staged = mktempdir(dirname(destination))
    backup = nothing
    try
        for (relative, contents) in files
            staged_file = joinpath(staged, relative)
            mkpath(dirname(staged_file))
            write(staged_file, contents)
        end
        if isdir(destination)
            backup = tempname(dirname(destination))
            mv(destination, backup)
        end
        try
            mv(staged, destination)
        catch
            backup === nothing || mv(backup, destination)
            rethrow()
        end
        backup === nothing || rm(backup; recursive=true)
    finally
        isdir(staged) && rm(staged; recursive=true)
    end
    for (relative, contents) in changed
        startswith(relative, snapshot_path * "/") && continue
        destination = joinpath(root, relative)
        mkpath(dirname(destination))
        temporary, stream = mktemp(dirname(destination))
        try
            write(stream, contents)
            close(stream)
            mv(temporary, destination; force=true)
        finally
            isopen(stream) && close(stream)
            isfile(temporary) && rm(temporary)
        end
    end
    return "published"
end

function main(args)
    options = Dict{String, String}()
    dry_run = false
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--dry-run"
            dry_run && error("Duplicate --dry-run")
            dry_run = true
        else
            argument in ("--report", "--collection", "--site", "--mode", "--expected-sha", "--expected-tag") || error("Unknown argument: $argument")
            haskey(options, argument) && error("Duplicate argument: $argument")
            index < length(args) || error("Missing value for $argument")
            index += 1
            options[argument] = args[index]
        end
        index += 1
    end
    all(haskey(options, key) for key in ("--site", "--mode")) && xor(haskey(options, "--report"), haskey(options, "--collection")) || error("Usage: publish.jl --report PATH | --collection PATH --site PATH --mode dev|main|release [--dry-run] [--expected-sha SHA] [--expected-tag TAG]")
    publisher, input = haskey(options, "--collection") ? (publish_collection, options["--collection"]) : (publish_report, options["--report"])
    status = publisher(input, options["--site"]; mode=options["--mode"], dry_run,
                            expected_sha=get(options, "--expected-sha", nothing), expected_tag=get(options, "--expected-tag", nothing))
    println(status)
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PerformancePublisher.main(ARGS)
end
