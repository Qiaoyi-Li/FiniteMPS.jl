using Test, JSON3
include("../metadata.jl")
include("../publish.jl")
include("../ci/resolve_target.jl")
include("../ci/check_run.jl")
include("../ci/publish_site.jl")

const TEST_ENVIRONMENT = PerformanceMetadata.collect_environment()
test_sha(n) = string(n; base=16, pad=40)

function test_report(n; tag=nothing, sha=test_sha(n), samples=1)
    Dict("schema_version"=>1,
        "source"=>Dict("repository"=>"Qiaoyi-Li/FiniteMPS.jl", "commit_sha"=>sha,
            "benchmark_source_sha"=>sha, "package_version"=>"0.0.0", "tag"=>tag, "prerelease"=>false),
        "run"=>Dict("measured_at_utc"=>"2026-09-29T00:00:00Z", "event_name"=>isnothing(tag) ? "push" : "release",
            "workflow"=>"Performance", "run_id"=>string(n), "run_number"=>n, "run_attempt"=>1, "run_url"=>nothing),
        "environment"=>TEST_ENVIRONMENT,
        "cases"=>[Dict("case_id"=>"test/publication", "description"=>"Synthetic publication fixture",
            "parameters"=>Dict(), "measurement_parameters"=>Dict("seed"=>1, "evals"=>1,
                "seconds_budget"=>1.0, "samples_budget"=>samples), "samples"=>samples,
            "median_time_ns"=>1.0, "allocated_bytes"=>0, "allocations"=>0)])
end

@testset "Branch reports and release archives" begin
    mktempdir() do directory
        site = mkpath(joinpath(directory,"site"))
        input = joinpath(directory,"report.json")
        root = joinpath(site,"performance")
        report_at(path) = JSON3.read(read(joinpath(root,path,"report.json"),String))
        function publish(mode,n; kwargs...)
            write(input,JSON3.write(test_report(n;kwargs...)))
            PerformancePublisher.publish_report(input,site;mode)
        end
        function push_branch(branch,n)
            selected = PerformanceTarget.select_target("push",
                Dict("ref"=>"refs/heads/$branch","after"=>test_sha(n)),test_sha(999),"")
            publish(selected["publish_mode"],n;sha=selected["target_ref"])
        end

        push_branch("dev",20)
        push_branch("main",10)
        @test (report_at("dev")["source"]["commit_sha"],report_at("main")["source"]["commit_sha"]) ==
            (test_sha(20),test_sha(10))

        tags = ("v1.1.0","v1.0.0","v2.0.0-rc1")
        for (n,tag) in enumerate(tags)
            publish("release",n;tag,samples=2)
        end
        archives = Dict(tag=>read(joinpath(root,"releases",tag,"report.json")) for tag in tags)
        push_branch("main",30)
        push_branch("dev",21)
        push_branch("main",9)
        push_branch("dev",19)
        @test (report_at("dev")["source"]["commit_sha"],report_at("main")["source"]["commit_sha"]) ==
            (test_sha(21),test_sha(30))
        @test all(read(joinpath(root,"releases",tag,"report.json")) == archives[tag] for tag in tags)
        @test occursin("href=\"../releases/v1.1.0/\"",read(joinpath(root,"stable","index.html"),String))
        index = read(joinpath(root,"index.html"),String)
        @test all(occursin("href=\"$branch/\"",index) for branch in ("dev","main"))

        # Exercise the CI entry points that accept the new publication mode.
        write(joinpath(directory,"run-status.json"),JSON3.write(Dict("status"=>"measured","case_count"=>1)))
        write(joinpath(directory,"report.md"),"Synthetic publication fixture\n")
        withenv("GITHUB_OUTPUT"=>joinpath(directory,"output"),"GITHUB_STEP_SUMMARY"=>joinpath(directory,"summary")) do
            PerformanceCompletion.main([directory,"--mode","main"])
        end
        PerformanceSite.parse_arguments(["--report",input,"--site",site,
            "--publisher",joinpath(@__DIR__,"..","publish.jl"),"--project",dirname(@__DIR__),
            "--mode","main","--expected-sha",test_sha(19)])
        summary = IOBuffer()
        PerformanceSite.PerformanceSummary.write_summary(summary,
            PerformanceSite.PerformanceSummary.read_summary(input);publishing="published",published_mode="main")
        @test occursin("performance/main)",String(take!(summary)))
    end
end

include("../harness.jl")

module SweepSuiteTest
using BenchmarkTools, Random
using ..BenchmarkHarness
include("../benchmarks.jl")

const created = Tuple{Tuple{String,Int},Vector{String}}[]
function PerformanceFixtures.build_fixture(rng::Xoshiro, model::NamedTuple, kind::String, D::Int)
    env = String[]
    push!(created,((kind,D),env))
    return (;env,parameters=Dict())
end
sweep!(env::Vector{String},operation,D) = push!(env,operation)
end

@testset "Single-sample sweeps with staggered dimensions" begin
    cases = filter(case->case.parameters["model"]=="hubbard_u1u1",
        SweepSuiteTest.benchmark_suite(BenchmarkHarness.ExecutionConfig()))
    results = BenchmarkHarness.measure_suite(cases; progress=devnull)
    @test all(case["samples"]==1 for case in results)
    @test SweepSuiteTest.created == [
        (("ground",128),["2-DMRG","2-DMRG"]),
        (("ground",256),["2-DMRG","CBE-DMRG","CBE-DMRG"]),
        (("ground",512),["2-DMRG","CBE-DMRG"]),
        (("ground",1024),["CBE-DMRG"]),
        (("thermal",128),["2-TDVP","2-TDVP"]),
        (("thermal",256),["2-TDVP","CBE-TDVP","CBE-TDVP"]),
        (("thermal",512),["2-TDVP","CBE-TDVP"]),
        (("thermal",1024),["CBE-TDVP"])]
    @test [(case.parameters["cbe_target"],case.parameters["continuation_after"])
        for case in cases if case.parameters["nominal_D"]==1024] == [(2048,nothing),(1152,nothing)]
    report = test_report(1)
    report["cases"] = results
    BenchmarkHarness.PerformanceReports.validate_report(report)
end
