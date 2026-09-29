"Resolve an Actions event to one immutable benchmark commit without shell evaluation."
module PerformanceTarget

using Downloads
using JSON3

export version_tag, select_target, resolve, release_prerelease

const SHA = r"^[0-9a-fA-F]{40}\z"
const SEMVER = r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?\z"

function version_tag(value)::Bool
    value isa AbstractString || return false
    matched = match(SEMVER, value)
    isnothing(matched) && return false
    prerelease = matched.captures[4]
    isnothing(prerelease) && return true
    return all(split(prerelease, '.')) do part
        !(all(isdigit, part) && length(part) > 1 && startswith(part, "0"))
    end
end

is_prerelease(tag::AbstractString) = !isnothing(match(SEMVER, tag).captures[4])
is_sha(value) = value isa AbstractString && occursin(SHA, value)

function skipped(reason::AbstractString)
    return Dict{String,Any}("should_run" => false, "reason" => reason, "target_sha" => "",
        "tag" => "", "prerelease" => false, "publish_mode" => "none")
end

"Pure event policy; Git resolution is separate so it can be tested with local repositories."
function select_target(event_name::AbstractString, event::AbstractDict,
                       context_sha::AbstractString, context_ref::AbstractString)
    result = Dict{String,Any}("should_run" => true, "reason" => "", "tag" => "",
        "prerelease" => false, "publish_mode" => "none")
    if event_name == "push"
        get(event, "deleted", false) && return skipped("Deleted refs are not benchmarked.")
        ref = get(event, "ref", context_ref)
        if ref in ("refs/heads/dev", "refs/heads/main")
            sha = get(event, "after", context_sha)
            is_sha(sha) && sha != repeat("0", 40) ||
                error("A branch push requires a nonzero full event commit SHA.")
            return merge(result, Dict("target_ref" => sha, "publish_mode" => last(split(ref, '/'))))
        end
        if ref isa AbstractString && startswith(ref, "refs/tags/")
            tag = replace(ref, r"^refs/tags/" => ""; count=1)
            version_tag(tag) || return skipped("Tag is not a supported vMAJOR.MINOR.PATCH SemVer tag.")
            after = get(event, "after", context_sha)
            is_sha(after) && after != repeat("0", 40) ||
                error("A version tag push requires a nonzero full event object SHA.")
            return merge(result, Dict("target_ref" => ref, "event_object" => after, "tag" => tag,
                "prerelease" => is_prerelease(tag), "publish_mode" => "release"))
        end
        return skipped("Only dev, main and version tag pushes are supported.")
    elseif event_name == "release"
        release = get(event, "release", Dict{String,Any}())
        if get(event, "action", nothing) != "published" || get(release, "draft", false)
            return skipped("Only published, non-draft releases are benchmarked.")
        end
        tag = get(release, "tag_name", nothing)
        version_tag(tag) || return skipped("Release tag is not a supported SemVer tag.")
        return merge(result, Dict("target_ref" => "refs/tags/" * tag, "tag" => tag,
            "prerelease" => (get(release, "prerelease", false) || is_prerelease(tag)),
            "publish_mode" => "release"))
    elseif event_name == "workflow_dispatch"
        inputs = get(event, "inputs", Dict{String,Any}())
        mode = get(inputs, "mode", "run-only")
        ref = get(inputs, "ref", "dev")
        if mode == "archive-release"
            version_tag(ref) ||
                error("archive-release requires a version tag such as v1.2.3, not a branch or SHA.")
            return merge(result, Dict("target_ref" => "refs/tags/" * ref, "tag" => ref,
                "prerelease" => is_prerelease(ref), "publish_mode" => "release"))
        end
        mode == "run-only" || error("Manual mode must be run-only or archive-release.")
        # Branches, tags and SHAs are passed as Git argv, never shell source.
        ref isa AbstractString && occursin(r"^[A-Za-z0-9][A-Za-z0-9._/+\-]*\z", ref) ||
            error("Manual ref must be a branch, tag or full SHA.")
        if occursin("..", ref) || occursin("//", ref) || any(suffix -> endswith(ref, suffix), ("/", ".", ".lock"))
            error("Manual ref is not a valid Git reference.")
        end
        return merge(result, Dict("target_ref" => ref))
    end
    return skipped("Unsupported event.")
end

function git(repo::AbstractString, args::AbstractString...)
    command = Cmd(["git", "-C", String(repo), String.(args)...])
    return strip(read(pipeline(command; stderr=devnull), String))
end

function resolve(selected::AbstractDict, repo::AbstractString)
    selected["should_run"] || return selected
    result = Dict{String,Any}(selected)
    ref = pop!(result, "target_ref")
    # Fetch the exact selected ref, including the event SHA for branch pushes.
    git(repo, "fetch", "--no-tags", "origin", ref)
    object_sha = git(repo, "rev-parse", "--verify", "FETCH_HEAD")
    event_object = pop!(result, "event_object", nothing)
    sha = git(repo, "rev-parse", "--verify", "FETCH_HEAD^{commit}")
    # Accept a pushed tag's raw object identity or its peeled commit identity.
    if !isnothing(event_object) && lowercase(event_object) ∉ (lowercase(object_sha), lowercase(sha))
        error("Version tag moved after the push event; refusing to measure different code.")
    end
    is_sha(sha) || error("Resolved target is not a full Git commit SHA.")
    if is_sha(ref) && lowercase(sha) != lowercase(ref)
        error("Resolved commit differs from the requested event SHA.")
    end
    result["target_sha"] = sha
    return result
end

function escape_path_segment(value::AbstractString)
    output = IOBuffer()
    for byte in codeunits(value)
        if byte in UInt8('A'):UInt8('Z') || byte in UInt8('a'):UInt8('z') ||
           byte in UInt8('0'):UInt8('9') || byte in codeunits("-._~")
            write(output, byte)
        else
            print(output, '%', uppercase(string(byte; base=16, pad=2)))
        end
    end
    return String(take!(output))
end

"HTTP adapter returning (status, JSON body); error responses need no JSON parsing.
A network failure returns the sentinel (0, nothing) so callers can fall back rather than abort."
function default_request_json(url::AbstractString, headers::AbstractDict)
    output = IOBuffer()
    response = try
        Downloads.request(url; headers, output, timeout=30, throw=false)
    catch
        return 0, nothing
    end
    response isa Downloads.Response || return 0, nothing
    body = 200 <= response.status < 300 ? JSON3.read(String(take!(output)), Dict{String,Any}) : Dict{String,Any}()
    return response.status, body
end

"Read an available GitHub Release prerelease flag; request_json is injectable for offline tests.
Returns `nothing` when the release metadata cannot be read, so measurement never aborts."
function release_prerelease(repository::AbstractString, tag::AbstractString;
                            request_json=default_request_json)::Union{Bool,Nothing}
    occursin(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\z", repository) || error("Invalid GitHub repository name")
    headers = Dict("Accept" => "application/vnd.github+json", "User-Agent" => "FiniteMPS-performance")
    token = get(ENV, "GITHUB_TOKEN", "")
    isempty(token) || (headers["Authorization"] = "Bearer " * token)
    url = "https://api.github.com/repos/" * repository * "/releases/tags/" * escape_path_segment(tag)
    status, release = try
        request_json(url, headers)
    catch exception
        @warn "Cannot read GitHub Release metadata; keeping the tag-derived prerelease flag." exception
        return nothing
    end
    status == 404 && return false
    200 <= status < 300 || begin
        @warn "Cannot read GitHub Release prerelease flag (HTTP $status); keeping the tag-derived prerelease flag."
        return nothing
    end
    return get(release, "prerelease", false)
end

function main(args=ARGS)
    length(args) == 2 && args[1] == "--repo" || error("usage: resolve_target.jl --repo REPOSITORY_DIRECTORY")
    event = JSON3.read(read(ENV["GITHUB_EVENT_PATH"], String), Dict{String,Any})
    event_name = ENV["GITHUB_EVENT_NAME"]
    result = resolve(select_target(event_name, event, get(ENV, "GITHUB_SHA", ""),
                                   get(ENV, "GITHUB_REF", "")), args[2])
    if result["should_run"] && !isempty(result["tag"]) && event_name != "release" && !result["prerelease"]
        api = release_prerelease(ENV["GITHUB_REPOSITORY"], result["tag"])
        isnothing(api) || (result["prerelease"] = api)
    end
    output = get(ENV, "GITHUB_OUTPUT", "")
    if !isempty(output)
        open(output, "a") do stream
            for name in sort!(collect(keys(result)))
                println(stream, name, '=', result[name])
            end
        end
    end
    println(JSON3.write(result))
    summary = get(ENV, "GITHUB_STEP_SUMMARY", "")
    if !result["should_run"] && !isempty(summary)
        open(summary, "a") do stream
            println(stream, "Performance run skipped: ", result["reason"])
        end
    end
    return result
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    try
        PerformanceTarget.main()
    catch error
        print(stderr, "Target resolution failed: ")
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
