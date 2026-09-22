include("collection_reports.jl")
using .PerformanceCollections

function main(args)
    iseven(length(args)) || error("Usage: build.jl --collection PATH [--output DIR] or --report PATH [--output DIR]")
    options = Dict{String, String}()
    for index in 1:2:length(args)
        flag = args[index]
        flag in ("--collection", "--report", "--output") || error("Unknown option: $flag")
        haskey(options, flag) && error("Duplicate option: $flag")
        options[flag] = args[index+1]
    end
    xor(haskey(options, "--collection"), haskey(options, "--report")) || error("Choose exactly one collection or report")
    if haskey(options, "--collection")
        input = options["--collection"]
        println(build_collection(input; output=get(options, "--output", dirname(abspath(input)))))
    else
        input = options["--report"]
        reports = PerformanceCollections.PerformanceReports
        report = reports.read_report(input)
        output = get(options, "--output", dirname(abspath(input)))
        files = Dict("report.json" => reports.json_text(report), "index.html" => reports.render_report(report),
                     "report.md" => reports.render_markdown(report))
        println(PerformanceCollections.write_build_files(files, output))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
