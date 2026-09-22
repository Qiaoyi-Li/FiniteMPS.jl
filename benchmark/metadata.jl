module PerformanceMetadata

using LinearAlgebra

export collect_environment, parse_cpu_list, runtime_configuration

"""Parse a Linux CPU list such as `0-3,8,10-11` into sorted, unique CPU IDs."""
function parse_cpu_list(value::AbstractString)::Vector{Int}
    isempty(strip(value)) && throw(ArgumentError("CPU list must not be empty"))
    cpus = Int[]
    for entry in split(strip(value), ','; keepempty=true)
        match_entry = match(r"^(\d+)(?:-(\d+))?$", strip(entry))
        isnothing(match_entry) && throw(ArgumentError("Invalid CPU list entry: $(repr(entry))"))
        first_cpu = tryparse(Int, match_entry.captures[1])
        last_cpu = isnothing(match_entry.captures[2]) ? first_cpu :
                   tryparse(Int, match_entry.captures[2])
        if isnothing(first_cpu) || isnothing(last_cpu) || last_cpu < first_cpu
            throw(ArgumentError("Invalid CPU list range: $(repr(entry))"))
        end
        append!(cpus, first_cpu:last_cpu)
    end
    return sort!(unique!(cpus))
end

# Hardware details are optional. Missing guest topology or affinity must not
# prevent a successful benchmark; required Julia/BLAS configuration is read below
# without this fallback so an API/configuration error cannot become a fake value.
function optional_read(f)
    try
        return f()
    catch
        return nothing
    end
end

optional_text(value) = isnothing(value) || isempty(strip(value)) ? nothing : String(value)

function process_affinity()
    Sys.islinux() || return (nothing, nothing)
    value = optional_read() do
        status = read("/proc/self/status", String)
        entry = match(r"(?m)^Cpus_allowed_list:\s*([^\n]+)$", status)
        isnothing(entry) && return nothing
        cpu_list = strip(entry.captures[1])
        return (cpu_list, length(parse_cpu_list(cpu_list)))
    end
    return isnothing(value) ? (nothing, nothing) : value
end

function visible_physical_cores()
    Sys.islinux() || return nothing
    return optional_read() do
        cpu_root = "/sys/devices/system/cpu"
        online_cpus = parse_cpu_list(read(joinpath(cpu_root, "online"), String))
        topology = Set{Tuple{Int,Int}}()
        for cpu in online_cpus
            directory = joinpath(cpu_root, "cpu$cpu", "topology")
            socket = parse(Int, strip(read(joinpath(directory, "physical_package_id"), String)))
            core = parse(Int, strip(read(joinpath(directory, "core_id"), String)))
            # A negative topology ID means that the kernel did not expose it.
            (socket < 0 || core < 0) && return nothing
            push!(topology, (socket, core))
        end
        return length(topology)
    end
end

"""Read the current Julia process's actual thread pools and loaded BLAS settings."""
function runtime_configuration()::Dict
    return Dict{String,Any}(
        "julia_version" => string(VERSION),
        "julia_threads_default" => Threads.nthreads(:default),
        "julia_threads_interactive" => Threads.nthreads(:interactive),
        "julia_gc_threads" => Threads.ngcthreads(),
        "blas_configuration" => sprint(show, BLAS.get_config()),
        "blas_threads" => BLAS.get_num_threads(),
    )
end

"""
    collect_environment(; runner_label=get(ENV, "PERFORMANCE_RUNNER_LABEL", nothing), packages=Module[])

Collect only the report's whitelisted environment fields in the measuring
process, after dependencies and thread settings have been initialized. CPU
counts describe resources visible to the operating system; affinity describes
allowed scheduling, and neither reports CPU utilization or exclusive host cores.
Unavailable optional fields use `nothing` (JSON `null`). Package versions come
from the supplied loaded modules; no package paths or full environment are saved.
"""
function collect_environment(;
    runner_label=get(ENV, "PERFORMANCE_RUNNER_LABEL", nothing),
    packages=Module[],
)::Dict
    cpu_info = Sys.cpu_info()
    affinity_list, affinity_count = process_affinity()
    package_versions = Dict{String,Any}()
    for package in packages
        version = Base.pkgversion(package)
        package_versions[string(nameof(package))] = isnothing(version) ? nothing : string(version)
    end
    kernel = Sys.isunix() ? optional_read(() -> strip(read(`uname -r`, String))) : nothing
    runner_environment = optional_text(get(ENV, "RUNNER_ENVIRONMENT", nothing))
    if runner_environment === nothing && get(ENV, "GITHUB_ACTIONS", "false") != "true"
        runner_environment = "local"
    end
    return Dict{String,Any}(
        "cpu" => Dict{String,Any}(
            "cpu_models" => unique([strip(cpu.model) for cpu in cpu_info]),
            "architecture" => string(Sys.ARCH),
            "logical_cpus_visible" => length(cpu_info),
            "physical_cores_visible" => visible_physical_cores(),
            "affinity_cpu_list" => affinity_list,
            "affinity_cpu_count" => affinity_count,
        ),
        "system" => Dict{String,Any}(
            "os" => string(Sys.KERNEL),
            "kernel" => kernel,
            "memory_total_bytes_visible" => Sys.total_memory(),
        ),
        "runner" => Dict{String,Any}(
            "label" => optional_text(runner_label),
            "environment" => runner_environment,
            "image_version" => optional_text(get(ENV, "ImageVersion", nothing)),
        ),
        "runtime" => runtime_configuration(),
        "package_versions" => package_versions,
    )
end

end # module PerformanceMetadata
