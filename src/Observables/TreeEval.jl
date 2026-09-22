abstract type TreeEvalAlgorithm end

struct LayeredTreeEval <: TreeEvalAlgorithm
    ntasks::Int
    function LayeredTreeEval(; ntasks = Threads.nthreads(:default))
        ntasks isa Integer && 1 <= ntasks <= Threads.nthreads(:default) ||
            throw(ArgumentError("ntasks must be an integer in 1:Threads.nthreads(:default)"))
        return new(ntasks)
    end
end

struct _TreeJoin
    left::Int
    right::Int
    channels::Vector{InteractionChannel}
end

struct _TreePlan
    nodes::Vector{InteractionTreeNode}
    parents::Vector{Int}
    sides::Vector{Bool}
    pushes::Vector{Vector{Int}}
    joins::Vector{Vector{_TreeJoin}}
    consumers::Vector{Int}
    maxwidth::Int
    total::Int
end

function _tree_plan(Tree)
    nodes = InteractionTreeNode[Tree.RootL, Tree.RootR]
    parents = [0, 0]
    sides = [true, false]
    depths = [0, 0]
    ids = IdDict(Tree.RootL => 1, Tree.RootR => 2)
    pushes = [Int[]]
    widths = [2]
    i = 1
    while i <= length(nodes)
        for child in nodes[i].children
            push!(nodes, child)
            push!(parents, i)
            push!(sides, sides[i])
            depth = depths[i] + 1
            push!(depths, depth)
            if length(pushes) <= depth
                push!(pushes, Int[])
                push!(widths, 0)
            end
            id = length(nodes)
            ids[child] = id
            push!(pushes[depth + 1], id)
            widths[depth + 1] += 1
        end
        i += 1
    end
    joins = [_TreeJoin[] for _ in pushes]
    consumers = map(node -> length(node.children), nodes)
    seen = IdDict{Any, Nothing}()
    njoin = 0
    for (left, node) in enumerate(nodes)
        sides[left] || continue
        groups = Dict{Int, Vector{InteractionChannel}}()
        for channel in node.Intrs
            isempty(channel.Ops) || error("tree channel has unexpanded operators")
            channel.LeafL === node || error("invalid left channel endpoint")
            right = ids[channel.LeafR]
            node.Op[1] + 1 == channel.LeafR.Op[1] || error("nonadjacent tree endpoints")
            push!(get!(groups, right, InteractionChannel[]), channel)
            haskey(seen, channel.ref) && error("result reference belongs to multiple channels")
            seen[channel.ref] = nothing
        end
        for (right, channels) in groups
            round = max(depths[left], depths[right]) + 1
            push!(joins[round], _TreeJoin(left, right, channels))
            consumers[left] += 1
            consumers[right] += 1
            njoin += 1
        end
    end
    for targets in values(Tree.Refs), ref in values(targets)
        haskey(seen, ref) || error("result reference has no tree channel")
    end
    return _TreePlan(nodes, parents, sides, pushes, joins, consumers,
        maximum(widths), length(nodes) - 2 + njoin)
end

mutable struct _TreeEnvEntry
    value::Any
    remaining::Int
    borrowers::Int
    cached::Bool
    file::Bool
    io::Symbol
    previous::Int
    next::Int
    condition::Threads.Condition
end

mutable struct _TreeEnvStore
    entries::Vector{_TreeEnvEntry}
    lock::ReentrantLock
    directory::String
    capacity::Int
    first::Int
    last::Int
    size::Int
    failure::Any
end

function _TreeEnvStore(consumers, disk::Bool, capacity::Int)
    guard = ReentrantLock()
    entries = [_TreeEnvEntry(nothing, n, 0, false, false, :none, 0, 0,
        Threads.Condition(guard)) for n in consumers]
    return _TreeEnvStore(entries, guard, disk ? mktempdir() : "", capacity, 0, 0, 0, nothing)
end

_tree_filename(store::_TreeEnvStore, id::Int) = joinpath(store.directory, "$(id).bin")

function _tree_uncache!(store, id)
    entry = store.entries[id]
    entry.cached || return nothing
    entry.previous == 0 ? (store.first = entry.next) :
        (store.entries[entry.previous].next = entry.next)
    entry.next == 0 ? (store.last = entry.previous) :
        (store.entries[entry.next].previous = entry.previous)
    entry.previous = entry.next = 0
    entry.cached = false
    store.size -= 1
    return nothing
end

function _tree_admit!(store, id)
    entry = store.entries[id]
    (entry.cached || entry.remaining == 0 || store.capacity == 0) && return 0
    victim = store.size == store.capacity ? store.first : 0
    victim != 0 && _tree_uncache!(store, victim)
    entry.previous = store.last
    entry.next = 0
    store.last == 0 ? (store.first = id) : (store.entries[store.last].next = id)
    store.last = id
    entry.cached = true
    store.size += 1
    return victim
end

function _tree_fail!(store, err)
    lock(store.lock) do
        if isnothing(store.failure)
            store.failure = err
            for entry in store.entries
                notify(entry.condition; all = true)
            end
        end
    end
    return nothing
end

function _tree_settle!(store, id)
    id == 0 && return nothing
    entry = store.entries[id]
    action, value = lock(store.lock) do
        (!isnothing(store.failure) || entry.borrowers != 0 || entry.io != :none) && return (:none, nothing)
        if entry.remaining == 0
            _tree_uncache!(store, id)
            entry.value = nothing
            if entry.file
                entry.io = :deleting
                return (:delete, nothing)
            end
        elseif !isempty(store.directory) && !entry.cached && !isnothing(entry.value)
            if entry.file
                entry.value = nothing
            else
                entry.io = :writing
                return (:write, entry.value)
            end
        end
        return (:none, nothing)
    end
    action == :none && return nothing
    try
        if action == :write
            serialize(_tree_filename(store, id), value)
        else
            rm(_tree_filename(store, id); force = true)
        end
        lock(store.lock) do
            entry.file = action == :write
            entry.io = :none
            notify(entry.condition; all = true)
        end
    catch err
        lock(store.lock) do
            entry.io = :none
        end
        _tree_fail!(store, err)
        rethrow()
    end
    return _tree_settle!(store, id)
end

function _tree_publish!(store, id, value)
    victim = lock(store.lock) do
        isnothing(store.failure) || throw(store.failure)
        entry = store.entries[id]
        entry.value = value
        return isempty(store.directory) ? 0 : _tree_admit!(store, id)
    end
    _tree_settle!(store, victim)
    _tree_settle!(store, id)
    return nothing
end

function _tree_borrow!(store, id)
    entry = store.entries[id]
    lock(store.lock)
    registered = false
    try
        isnothing(store.failure) || throw(store.failure)
        entry.borrowers += 1
        registered = true
        while entry.io == :reading
            wait(entry.condition)
            isnothing(store.failure) || throw(store.failure)
        end
        if !isnothing(entry.value)
            return entry.value
        end
        entry.file || error("environment $id is unavailable")
        entry.io = :reading
    catch
        registered && (entry.borrowers -= 1)
        rethrow()
    finally
        unlock(store.lock)
    end
    try
        value = deserialize(_tree_filename(store, id))
        victim = lock(store.lock) do
            entry.value = value
            entry.io = :none
            victim = _tree_admit!(store, id)
            notify(entry.condition; all = true)
            return victim
        end
        _tree_settle!(store, victim)
        return value
    catch err
        lock(store.lock) do
            entry.borrowers -= 1
            entry.io = :none
        end
        _tree_fail!(store, err)
        rethrow()
    end
end

function _tree_return!(store, id; consumed::Bool = false)
    lock(store.lock) do
        entry = store.entries[id]
        entry.borrowers -= 1
        consumed && (entry.remaining -= 1)
    end
    return _tree_settle!(store, id)
end

function _tree_phase!(f, jobs, store, ntasks, timer)
    isempty(jobs) && return 0
    next = Ref(1)
    done = fill(false, length(jobs))
    timers = [TimerOutput() for _ in 1:min(ntasks, length(jobs))]
    function worker(localtimer)
        while true
            i = lock(store.lock) do
                (!isnothing(store.failure) || next[] > length(jobs)) && return 0
                i = next[]
                next[] += 1
                return i
            end
            i == 0 && return nothing
            try
                f(jobs[i], localtimer)
                done[i] = true
            catch err
                _tree_fail!(store, err)
                return nothing
            end
        end
    end
    if ntasks == 1
        worker(only(timers))
    else
        @sync for localtimer in timers
            Threads.@spawn worker(localtimer)
        end
    end
    for localtimer in timers
        merge!(timer, localtimer; tree_point = String[])
    end
    isnothing(store.failure) || throw(store.failure)
    all(done) || error("incomplete tree stage")
    return length(jobs)
end

function _tree_options(kwargs, maxsize, GCspacing, showtimes)
    for key in keys(kwargs)
        key == :serial && throw(ArgumentError("serial was removed; use alg=LayeredTreeEval(ntasks=1) for serial execution"))
        key == :ntasks && throw(ArgumentError("use alg=LayeredTreeEval(ntasks=N) instead of ntasks=N"))
        key == :maxdegree && throw(ArgumentError("maxdegree was removed; use maxsize for the shared environment cache"))
        throw(ArgumentError("unsupported tree evaluation keyword: $key"))
    end
    isnothing(maxsize) || (maxsize isa Integer && maxsize >= 0) ||
        throw(ArgumentError("maxsize must be a nonnegative integer"))
    GCspacing isa Integer && GCspacing >= 0 || throw(ArgumentError("GCspacing must be a nonnegative integer"))
    showtimes isa Integer && showtimes > 0 || throw(ArgumentError("showtimes must be a positive integer"))
    return nothing
end

function _evaluate_tree!(Tree, prepare, pushenv, left, right, alg::LayeredTreeEval;
    disk::Bool, maxsize, verbose::Integer, showtimes::Integer, GCspacing::Integer)
    plan = _tree_plan(Tree)
    timer = TimerOutput()
    store = _TreeEnvStore(plan.consumers, disk, isnothing(maxsize) ? plan.maxwidth : Int(maxsize))
    completed = 0
    sincegc = 0
    spacing = max(1, cld(plan.total, showtimes))
    nextshow = spacing
    function barrier(n)
        n == 0 && return nothing
        completed += n
        sincegc += n
        if GCspacing > 0 && sincegc >= GCspacing
            @timeit timer "GC" GC.gc()
            sincegc = 0
        end
        if verbose > 0 && (completed >= nextshow || completed == plan.total)
            show(timer; title = "$completed / $(plan.total)")
            println()
            flush(stdout)
            nextshow = completed + spacing
        end
        return nothing
    end
    try
        _tree_publish!(store, 1, left)
        _tree_publish!(store, 2, right)
        for round in eachindex(plan.pushes)
            jobs = plan.pushes[round]
            leftjob = findfirst(id -> plan.sides[id], jobs)
            rightjob = findfirst(id -> !plan.sides[id], jobs)
            leftdata = isnothing(leftjob) ? nothing : prepare(true, plan.nodes[jobs[leftjob]].Op[1])
            rightdata = isnothing(rightjob) ? nothing : prepare(false, plan.nodes[jobs[rightjob]].Op[1])
            n = _tree_phase!(jobs, store, alg.ntasks, timer) do id, localtimer
                parent = plan.parents[id]
                @timeit localtimer "load environment" env = _tree_borrow!(store, parent)
                consumed = false
                try
                    side = plan.sides[id]
                    @timeit localtimer "push" result = pushenv(side, plan.nodes[id], env,
                        side ? leftdata : rightdata)
                    @timeit localtimer "store environment" _tree_publish!(store, id, result)
                    consumed = true
                finally
                    @timeit localtimer "release environment" _tree_return!(store, parent; consumed)
                end
            end
            barrier(n)
            n = _tree_phase!(plan.joins[round], store, alg.ntasks, timer) do join, localtimer
                @timeit localtimer "load environment" El = _tree_borrow!(store, join.left)
                consumed = false
                try
                    @timeit localtimer "load environment" Er = _tree_borrow!(store, join.right)
                    try
                        @timeit localtimer "join" value = El * Er
                        for channel in join.channels
                            channel.ref[] = channel.phase * value
                        end
                        consumed = true
                    finally
                        @timeit localtimer "release environment" _tree_return!(store, join.right; consumed)
                    end
                finally
                    @timeit localtimer "release environment" _tree_return!(store, join.left; consumed)
                end
            end
            barrier(n)
        end
    finally
        for entry in store.entries
            entry.value = nothing
        end
        isempty(store.directory) || rm(store.directory; recursive = true, force = true)
    end
    return timer
end
