module PerformanceCollectionRunner

using JSON3
include("collection_reports.jl")
using .PerformanceCollections

export collect_reports, configuration_command

write_json(path, value) = open(path,"w") do io
    JSON3.pretty(io,value)
    println(io)
end
read_json(path) = JSON3.read(read(path,String),Dict{String,Any})

function configuration_command(n, output; suite=nothing,
                               expected_sha=nothing, tag=nothing, prerelease=false,
                               samples=nothing, seconds=nothing)
    args = ["--output",abspath(output),"--julia-threads",string(n),
            "--blas-threads","1","--gc-threads","1"]
    for (flag,value) in (("--suite",suite),("--expected-sha",expected_sha),("--tag",tag),
                         ("--samples",samples),("--seconds",seconds))
        isnothing(value) || append!(args,[flag,string(value)])
    end
    append!(args,["--prerelease",string(prerelease)])
    return `$(Base.julia_cmd()) --startup-file=no --threads=$n --gcthreads=1 --project=$(@__DIR__) $(joinpath(@__DIR__,"run.jl")) $args`
end

"Run independent 1/2/4-thread subprocesses sequentially and publish a completion marker last."
function collect_reports(; output=joinpath(@__DIR__,"output"), suite=nothing,
                         expected_sha=nothing, tag=nothing, prerelease=false,
                         samples=nothing, seconds=nothing,
                         run_child=run,
                         build_site=(path,destination)->build_collection(path;output=destination),
                         progress=stderr)
    output=abspath(output)
    mkpath(output)
    completed=0
    top_level=("collection.json","index.html","report.json","report.md","run-status.json")
    for name in top_level
        path=joinpath(output,name)
        (ispath(path)||islink(path)) && rm(path;force=true)
    end
    # Remove only the collector's three known configuration output directories.
    for n in (1,2,4)
        path=joinpath(output,"configurations","julia-$n-blas-1")
        (ispath(path)||islink(path)) && rm(path;recursive=true,force=true)
    end
    try
        isnothing(samples) || samples>=2 || error("samples override must be at least 2")
        isnothing(seconds) || (isfinite(seconds)&&seconds>0) || error("seconds override must be positive and finite")
        reports=Dict{String,Any}[]
        configurations=Dict{String,Any}[]
        statuses=String[]
        for n in (1,2,4)
            relative="configurations/julia-$n-blas-1/report.json"
            directory=dirname(joinpath(output,relative))
            println(progress,"Collecting Julia threads=$n, BLAS threads=1, GC threads=1")
            flush(progress)
            command=configuration_command(n,directory;suite,expected_sha,tag,prerelease,samples,seconds)
            withenv("JULIA_NUM_THREADS"=>string(n),"JULIA_NUM_GC_THREADS"=>"1","OPENBLAS_NUM_THREADS"=>"1") do
                Base.invokelatest(run_child,command)
            end
            status=read_json(joinpath(directory,"run-status.json"))
            state=get(status,"status",nothing)
            count=get(status,"case_count",nothing)
            count isa Integer && !(count isa Bool) && count>=0 || error("invalid child case_count")
            isfile(joinpath(directory,"report.md")) || error("child did not produce its summary")
            if state=="measured"
                report=read_json(joinpath(output,relative))
                count>0 && length(report["cases"])==count || error("child report case count differs from status")
                push!(reports,report)
            elseif state=="empty"
                count==0 && !ispath(joinpath(output,relative)) && !islink(joinpath(output,relative)) || error("empty child contains performance data")
            else
                error("child did not complete successfully")
            end
            push!(statuses,state)
            push!(configurations,Dict("julia_threads"=>n,"blas_threads"=>1,"gc_threads"=>1,"report_path"=>relative))
            completed+=1
        end
        if all(==("empty"),statuses)
            write(joinpath(output,"report.md"),"# Performance\n\nNo benchmark cases configured in any of the Julia 1, 2, and 4 thread configurations. No performance report was measured or published.\n")
            write_json(joinpath(output,"run-status.json"),Dict("status"=>"empty","case_count"=>0,"configuration_count"=>3))
            return nothing
        end
        all(==("measured"),statuses) || error("configurations disagree on whether benchmark cases exist")
        index=Dict("schema_version"=>1,"kind"=>"performance_collection","status"=>"complete",
                   "source"=>deepcopy(first(reports)["source"]),"run"=>deepcopy(first(reports)["run"]),
                   "configurations"=>configurations)
        validate_collection(index,reports)
        collection_path=joinpath(output,"collection.json")
        write_json(collection_path,index)
        bundle=read_collection(collection_path)
        Base.invokelatest(build_site,collection_path,output)
        isfile(joinpath(output,"index.html")) || error("collection HTML was not generated")
        write_json(joinpath(output,"run-status.json"),Dict("status"=>"measured",
                   "case_count"=>length(first(reports)["cases"]),"configuration_count"=>3))
        return bundle
    catch
        # Child reports remain available for diagnosis; no partial collection is complete.
        for name in top_level
            path=joinpath(output,name)
            (ispath(path)||islink(path)) && rm(path;force=true)
        end
        write_json(joinpath(output,"run-status.json"),Dict("status"=>"failed","case_count"=>0,
                   "completed_configurations"=>completed))
        rethrow()
    end
end

function main(args=ARGS)
    options=Dict{String,String}()
    allowed=("--output","--expected-sha","--tag","--prerelease","--suite","--samples","--seconds")
    iseven(length(args)) || error("arguments must be --option value pairs")
    for i in 1:2:length(args)
        args[i] in allowed || error("unknown option: $(args[i])")
        haskey(options,args[i]) && error("duplicate option: $(args[i])")
        options[args[i]]=args[i+1]
    end
    get(options,"--prerelease","false") in ("true","false") || error("--prerelease must be true or false")
    return collect_reports(;output=get(options,"--output",joinpath(@__DIR__,"output")),
        suite=get(options,"--suite",nothing),
        expected_sha=get(options,"--expected-sha",nothing),tag=get(options,"--tag",nothing),
        prerelease=get(options,"--prerelease","false")=="true",
        samples=haskey(options,"--samples") ? parse(Int,options["--samples"]) : nothing,
        seconds=haskey(options,"--seconds") ? parse(Float64,options["--seconds"]) : nothing)
end

end
if abspath(PROGRAM_FILE)==@__FILE__
    PerformanceCollectionRunner.main()
end
