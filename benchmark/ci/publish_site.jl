"""Publish through the existing Documenter SSH deployment key.

Run only inside the shared gh-pages-publish job lock. This script, the publisher,
and their environment must come from the workflow control checkout.
"""
module PerformanceSite

# Remove the CLI secret before imports can launch Julia precompilation children.
# Including this module for tests or reuse must leave the caller's ENV alone.
const _cli_deploy_key = Ref{Union{Nothing, String}}(
    abspath(PROGRAM_FILE) == (@__FILE__) ? pop!(ENV, "DOCUMENTER_KEY", "") : nothing)

using Base64
using Downloads
using JSON3
include("summary.jl")
using .PerformanceSummary

export decode_key, prepare_site, commit_site

function git(site::AbstractString, args::AbstractString...; env=nothing)
    command = Cmd(["git", "-C", String(site), String.(args)...])
    if env === nothing
        return chomp(read(command, String))
    end
    # Inherit only through the process environment, so a failed Cmd does not
    # include a serialized environment (potentially containing secrets) in logs.
    return withenv(collect(pairs(env))...) do
        chomp(read(command, String))
    end
end

function decode_key(encoded::AbstractString)::Vector{UInt8}
    compact = replace(encoded, r"\s+" => "")
    occursin(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$", compact) ||
        error("DOCUMENTER_KEY must contain Documenter's base64-encoded SSH private key")
    key = try
        base64decode(compact)
    catch
        error("DOCUMENTER_KEY must contain Documenter's base64-encoded SSH private key")
    end
    occursin(r"^-----BEGIN (?:OPENSSH|RSA|EC|DSA|PRIVATE) (?:PRIVATE )?KEY-----\r?\n", String(copy(key))) ||
        error("Decoded DOCUMENTER_KEY is not an SSH private key")
    return key
end

"Read the latest branch only after acquiring the workflow's shared job lock."
function prepare_site(site::AbstractString, remote::AbstractString)
    (ispath(site) || islink(site)) && error("Site checkout must be a fresh directory")
    mkpath(site)
    git(site, "init", "--initial-branch=gh-pages")
    git(site, "remote", "add", "origin", remote)
    existing = git(site, "ls-remote", "--heads", "origin", "refs/heads/gh-pages")
    if !isempty(existing)
        git(site, "fetch", "--no-tags", "origin", "refs/heads/gh-pages")
        git(site, "checkout", "-B", "gh-pages", "FETCH_HEAD")
    end
    return nothing
end

function commit_site(site::AbstractString, identity::AbstractString)::Bool
    changed = git(site, "status", "--porcelain", "--untracked-files=all", "-z")
    isempty(changed) && return false
    entries = split(changed, '\0'; keepempty=false)
    index = 1
    while index <= length(entries)
        entry = entries[index]
        length(entry) >= 4 && startswith(entry[4:end], "performance/") ||
            error("Publisher changed a path outside performance/")
        # Porcelain -z represents staged renames/copies using two path entries.
        # Both paths must remain in the publisher's owned directory.
        if 'R' in entry[1:2] || 'C' in entry[1:2]
            index += 1
            index <= length(entries) && startswith(entries[index], "performance/") ||
                error("Publisher renamed or copied a path outside performance/")
        end
        index += 1
    end
    git(site, "add", "--", "performance")
    git(site, "-c", "user.name=github-actions[bot]", "-c",
        "user.email=41898282+github-actions[bot]@users.noreply.github.com",
        "commit", "-m", "Update performance report for $identity")
    return true
end

# GIT_SSH_COMMAND is interpreted by a shell, unlike the regular Cmd arguments.
shell_quote(value::AbstractString) = "'" * replace(value, "'" => "'\\''") * "'"

function github_host_keys()
    output = IOBuffer()
    response = Downloads.request("https://api.github.com/meta";
        headers=["User-Agent" => "FiniteMPS-performance"], output, timeout=30)
    response.status == 200 || error("GitHub SSH host-key metadata request failed")
    metadata = JSON3.read(String(take!(output)), Dict{String, Any})
    keys = get(metadata, "ssh_keys", nothing)
    keys isa AbstractVector && !isempty(keys) &&
        all(value -> value isa AbstractString && !isempty(value) && !occursin(r"[\r\n]", value), keys) ||
        error("GitHub returned invalid SSH host keys")
    return keys
end

function parse_arguments(args)
    options = Dict{String, String}()
    index = 1
    while index <= length(args)
        argument = args[index]
        argument in ("--report", "--collection", "--site", "--publisher", "--project", "--mode", "--expected-sha", "--expected-tag") ||
            error("Unknown argument: $argument")
        haskey(options, argument) && error("Duplicate argument: $argument")
        index < length(args) || error("Missing value for $argument")
        options[argument] = args[index + 1]
        index += 2
    end
    (haskey(options,"--report") ⊻ haskey(options,"--collection")) &&
        all(haskey(options, key) for key in ("--site", "--publisher", "--project", "--mode", "--expected-sha")) ||
        error("Usage: publish_site.jl (--report PATH | --collection PATH) --site PATH --publisher PATH --project PATH --mode dev|release --expected-sha SHA [--expected-tag TAG]")
    options["--mode"] in ("dev", "release") || error("Publishing mode must be dev or release")
    return options
end

function main(args=ARGS)
    encoded_key = _cli_deploy_key[]
    _cli_deploy_key[] = nothing
    encoded_key === nothing && (encoded_key = pop!(ENV, "DOCUMENTER_KEY", ""))
    options = parse_arguments(args)
    repository = ENV["GITHUB_REPOSITORY"]
    occursin(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", repository) || error("Invalid GitHub repository name")
    input_flag=haskey(options,"--collection") ? "--collection" : "--report"
    report_path, site = options[input_flag], options["--site"]
    report = JSON3.read(read(report_path, String), Dict{String, Any})
    source = get(report, "source", nothing)
    source isa AbstractDict && get(source, "repository", nothing) == repository ||
        error("Report source does not match the publishing repository")
    key = decode_key(encoded_key)
    # The deploy secret is removed before git reads or the child Julia renderer.
    prepare_site(site, "https://github.com/$repository.git")
    command = `$(Base.julia_cmd()) --startup-file=no --threads=2 --gcthreads=1 --project=$(options["--project"]) $(options["--publisher"]) $input_flag $report_path --site $site --mode $(options["--mode"]) --expected-sha $(options["--expected-sha"])`
    tag = get(options, "--expected-tag", "")
    isempty(tag) || (command = `$command --expected-tag $tag`)
    completed = read(command, String)
    print(completed)
    lines = split(strip(completed), '\n'; keepempty=false)
    isempty(lines) && error("Publisher did not return a recognized completion status")
    status = last(lines)
    status in ("published", "already-archived", "older-dev-skipped", "no-op") ||
        error("Publisher did not return a recognized completion status")
    changed = commit_site(site, isempty(tag) ? options["--expected-sha"] : tag)
    if changed
        # HTTPS authenticates GitHub's public host keys; do not accept blind
        # ssh-keyscan output. Only the final push receives the decoded key path.
        host_keys = github_host_keys()
        mktempdir(; prefix="performance-ssh-") do directory
            key_path, known_hosts = joinpath(directory, "key"), joinpath(directory, "known_hosts")
            open(key_path, "w") do stream
                chmod(key_path, 0o600)
                write(stream, key)
            end
            write(known_hosts, join(["github.com $value\n" for value in host_keys]))
            ssh_command = join(shell_quote.(["ssh", "-i", key_path, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=$known_hosts"]), " ")
            git(site, "push", "git@github.com:$repository.git", "HEAD:refs/heads/gh-pages";
                env=Dict("GIT_SSH_COMMAND" => ssh_command))
        end
    end
    open(ENV["GITHUB_STEP_SUMMARY"], "a") do stream
        write_summary(stream, read_summary(report_path); publishing=status,
                      git_push=changed, published_mode=options["--mode"])
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    PerformanceSite.main()
end
