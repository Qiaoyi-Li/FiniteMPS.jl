"""
     calITP!(G::ImagTimeProxyGraph, ρ::MPO; kwargs...) -> ::TimerOutput 

Calculate the values of ITP stroed in graph `G` and assign the results to `G.Refs`. Return the `TimerOutput` object which collects the time cost.

# Kwargs
     serial::Bool = false
Force to compute in serial mode, usually used for debugging.

     GCspacing::Int64 = 100
The spacing of manual garbage collection.

     verbose::Int64 = 0
Display the timer outputs during the calculation if `verbose > 0`.

     showtimes::Int64 = 100
The number of times to show the timer outputs.

     ntasks::Int64 = get_num_threads_julia() - 1
The number of tasks to be used in multi-threading mode.

     disk::Bool = false
Store the environment tensors in disk if `true`.

     maxdegree::Int64 = 4
This argument is only used if `disk=true`. The environment tensor in a vectex will be stored in disk only if its degree `< maxdegree`, otherwise it will be left in memory to avoid frequent disk I/O.
"""
function calITP!(G::ImagTimeProxyGraph, ρ::MPO{L}; kwargs...) where {L}

     # make sure a vertex is either in left or right tree
     _expand_left_tree!(G)

     # set traversal status for vertices and edges
     for v in vertices(G.graph)
          set_prop!(G.graph, v, :passed, false)
     end
     for e in edges(G.graph)
          if has_prop(G.graph, e, :Refs)
               set_prop!(G.graph, e, :passed, false)
          end
     end

     if get(kwargs, :serial, false)
          return _calITP_serial!(G, ρ; kwargs...)
     end

     if get_num_threads_julia() > 1
          return _calITP_threading!(G, ρ; kwargs...)
     else
          # fallback to serial
          return _calITP_serial!(G, ρ; kwargs...)
     end
end

function _calITP_serial!(G::ImagTimeProxyGraph, ρ::MPO{L}; kwargs...) where {L}

     GCspacing::Int64 = get(kwargs, :GCspacing, 100)
     verbose::Int64 = get(kwargs, :verbose, 0)
     showtimes::Int64 = get(kwargs, :showtimes, 100)

     # initialize the boundary environment
     set_prop!(G.graph, 1, :E, BilayerLeftTensor{1,1}(isometry(codomain(ρ[1])[1], codomain(ρ[1])[1])))
     set_prop!(G.graph, 1, :passed, true)
     set_prop!(G.graph, 2, :E, BilayerRightTensor{1,1}(isometry(domain(ρ[end])[end], domain(ρ[end])[end])))
     set_prop!(G.graph, 2, :passed, true)

     # print
     Timer_acc = TimerOutput()
     num_tot = nv(G.graph) - 2 # exclude the left and right roots
     num_count = 0
     showspacing::Int64 = cld(num_tot, showtimes)
     GC_count = 0

     # vertices to be calculated
     boundary_pool = Set{Int64}([1, 2])
     while !isempty(boundary_pool)
          v_next = argmin(x -> _priority_vertex(G, x), [x for v in boundary_pool for x in filter(children(G, v)) do ch
               # L tree can only 
               get_prop(G.graph, ch, :st) != get_prop(G.graph, v, :st) && return false
               return !get_prop(G.graph, ch, :passed)
          end])

          si = get_prop(G.graph, v_next, :si)
          _update_vertex!(G, v_next, ρ[si]', ρ[si], Timer_acc)

          # update boundary pool
          push!(boundary_pool, v_next)
          filter!(boundary_pool) do v
               has_prop(G.graph, v, :E)
          end

          num_count += 1
          # print 
          if verbose > 0 && iszero(num_count % showspacing)
               show(Timer_acc; title="$(num_count) / $(num_tot)")
               println()
               flush(stdout)
          end

          # manual GC
          GC_count += 1
          if GC_count == GCspacing
               GC_count = 0
               manualGC(Timer_acc)
          end
     end

     # one more GC to free the environment tensors 
     manualGC(Timer_acc)

     return Timer_acc
end

function _cond_clear(G::ImagTimeProxyGraph, v::Int64;
     v_pass::Union{Nothing,Int64}=nothing,
     e_pass::Union{Nothing,Edge}=nothing,
     edge_only::Bool=false)
     # check if the environment tensor in v can be cleared
     # if v_pass or e_pass is given, this vertex or edge will assume to be passed
     !get_prop(G.graph, v, :passed) && return false

     st = get_prop(G.graph, v, :st)
     if st == :L
          return all(children(G, v)) do child
               if get_prop(G.graph, child, :st) == :R
                    Edge(v, child) == e_pass && return true
                    return get_prop(G.graph, Edge(v, child), :passed)
               else
                    edge_only && return true
                    v_pass == child && return true
                    return get_prop(G.graph, child, :passed)
               end
          end
     else
          return all(children(G, v)) do child
               if get_prop(G.graph, child, :st) == :L
                    Edge(child, v) == e_pass && return true
                    return get_prop(G.graph, Edge(child, v), :passed)
               else
                    edge_only && return true
                    v_pass == child && return true
                    return get_prop(G.graph, child, :passed)
               end
          end
     end

end

function _priority_vertex(G::ImagTimeProxyGraph, v::Int64)
     # return the priority of a given vertex

     st = get_prop(G.graph, v, :st)

     v_parent = parent(G, v)
     v_children = children(G, v)

     # if the parent vertex can be removed after calculating this vertex
     if has_prop(G.graph, v_parent, :E) && _cond_clear(G, v_parent; v_pass=v)
          return 0
     end

     # if this vertex only has one child, so that the child has the highest priority and this vertex can be removed in the next iteration
     if length(v_children) == 1
          return 1
     end

     # increase the priority of leaf vertices so that the left and right leaf vertices can be earlier matched and removed
     if any(i -> get_prop(G.graph, i, :st) != st, v_children)
          return 2
     end

     # vertex with less children has higher priority to control the memory increase
     # 1 => 1, 2 => 3, 3 => 4, ...
     return length(v_children) + 1

end

function _update_vertex!(G::ImagTimeProxyGraph, v::Int64, A::AdjointMPSTensor, B::MPSTensor, LocalTimer::TimerOutput; kwargs...)

     if get_prop(G.graph, v, :st) == :L
          @timeit LocalTimer "update L vertex" _update_vertex_L!(G, v, A, B; kwargs...)
     else
          @timeit LocalTimer "update R vertex" _update_vertex_R!(G, v, A, B; kwargs...)
     end
     return nothing

end

function _update_vertex_L!(G::ImagTimeProxyGraph, v::Int64, A::AdjointMPSTensor, B::MPSTensor)
     si = get_prop(G.graph, v, :si)
     idx_Op = get_prop(G.graph, v, :idx_Op)
     O₁, O₂ = G.Ops[si][idx_Op]

     # compute El
     v_parent = parent(G, v)
     El = _pushright(get_prop(G.graph, v_parent, :E), A, O₁, B, O₂)
     set_prop!(G.graph, v, :E, El)
     set_prop!(G.graph, v, :passed, true)

     # check if the parent vertex can be cleared
     if _cond_clear(G, v_parent)
          delete!(G.graph.vprops[v_parent], :E)
     end

     # compute obs for leaf vertices
     for v_child in filter(x -> get_prop(G.graph, x, :st) == :R, children(G, v))
          if has_prop(G.graph, v_child, :E)
               # compute obs and update Refs on the edge
               Er = get_prop(G.graph, v_child, :E)
               val = El * Er
               for ref in get_prop(G.graph, Edge(v, v_child), :Refs)
                    ref[] = val
               end
               set_prop!(G.graph, Edge(v, v_child), :passed, true)
               # check if the child vertex can be cleared
               if _cond_clear(G, v_child)
                    delete!(G.graph.vprops[v_child], :E)
               end
          end
     end
     # check if current vertex can be cleared (e.g. leaf vertex with all edges passed)
     if _cond_clear(G, v)
          delete!(G.graph.vprops[v], :E)
     end

     return nothing

end

function _update_vertex_R!(G::ImagTimeProxyGraph, v::Int64, A::AdjointMPSTensor, B::MPSTensor)

     si = get_prop(G.graph, v, :si)
     idx_Op = get_prop(G.graph, v, :idx_Op)
     O₁, O₂ = G.Ops[si][idx_Op]

     # compute Er 
     v_parent = parent(G, v)
     Er = _pushleft(get_prop(G.graph, v_parent, :E), A, O₁, B, O₂)
     set_prop!(G.graph, v, :E, Er)
     set_prop!(G.graph, v, :passed, true)

     # check if the parent vertex can be cleared
     if _cond_clear(G, v_parent)
          delete!(G.graph.vprops[v_parent], :E)
     end

     # compute obs for leaf vertices
     for v_child in filter(x -> get_prop(G.graph, x, :st) == :L, children(G, v))
          if has_prop(G.graph, v_child, :E)
               # compute obs and update Refs on the edge
               El = get_prop(G.graph, v_child, :E)
               val = El * Er
               for ref in get_prop(G.graph, Edge(v_child, v), :Refs)
                    ref[] = val
               end
               set_prop!(G.graph, Edge(v_child, v), :passed, true)
               # check if the child vertex can be cleared
               if _cond_clear(G, v_child)
                    delete!(G.graph.vprops[v_child], :E)
               end
          end
     end
     # check if current vertex can be cleared (e.g. leaf vertex with all edges passed)
     if _cond_clear(G, v)
          delete!(G.graph.vprops[v], :E)
     end

     return nothing

end

function _calITP_threading!(G::ImagTimeProxyGraph, ρ::MPO{L}; kwargs...) where {L}

     ntasks::Int64 = get(kwargs, :ntasks, get_num_threads_julia() - 1)
     @assert ntasks ≤ get_num_threads_julia() - 1
     GCspacing::Int64 = get(kwargs, :GCspacing, 100)
     verbose::Int64 = get(kwargs, :verbose, 0)
     showtimes::Int64 = get(kwargs, :showtimes, 100)
     disk::Bool = get(kwargs, :disk, false)
     maxdegree::Int64 = get(kwargs, :maxdegree, 4)
     if disk
          # set up a temporary directory for storing the environment tensors
          tmppath = mktempdir()

          # still store E in memory for some high-degree vertices
          for v in vertices(G.graph)
               st = get_prop(G.graph, v, :st)
               d = mapreduce(+, children(G, v)) do x
                    # only count the opposite children 
                    get_prop(G.graph, x, :st) == st ? 0 : 1
               end
               # max degree for storing E in disk
               set_prop!(G.graph, v, :E, d ≤ maxdegree ? joinpath(tmppath, "E$(v).bin") : nothing)
          end
     else
          for v in vertices(G.graph)
               set_prop!(G.graph, v, :E, nothing)
          end
     end

     num_tot = nv(G.graph) + mapreduce(+, edges(G.graph)) do e
          has_prop(G.graph, e, :Refs) ? 1 : 0
     end

     Ch = Channel{Union{Int64,NTuple{2,Int64}}}(num_tot) # vertex, edge
     Ch_Timer = Channel{TimerOutput}(num_tot)

     Lock = ReentrantLock()
     # workers
     tasks_c = map(1:ntasks) do _
          Threads.@spawn while isopen(Ch)
               _calITP_worker!(Ch, Ch_Timer, G, ρ, Lock)
          end
     end

     # initialize the recursion
     put!(Ch, 1)
     put!(Ch, 2)

     # main thread
     Timer_acc = TimerOutput()
     showspacing::Int64 = cld(num_tot, showtimes)
     num_count = GC_count = 0
     while num_count < num_tot
          # check and rethrow if any task failed
          for task in tasks_c
               istaskfailed(task) && fetch(task)
          end
          to = take!(Ch_Timer)
          merge!(Timer_acc, to)
          num_count += 1
          GC_count += 1
          if GC_count == GCspacing
               GC_count = 0
               manualGC(Timer_acc)
          end
          if verbose > 0 && (iszero(num_count % showspacing) || num_count == num_tot) 
               show(Timer_acc; title="ITP $(num_count) / $(num_tot)")
               println()
               flush(stdout)
          end
     end

     # kill tasks
     close(Ch)
     close(Ch_Timer)

     # final clear
     @timeit Timer_acc "clear" begin
          for v in vertices(G.graph)
               if has_prop(G.graph, v, :E)
                    _clear_Env!(G, v)
               end
          end
     end
     disk && rm(tmppath; recursive=true)
     
     return Timer_acc
end

_pushEnv(E::BilayerLeftTensor, args...) = _pushright(E, args...)
_pushEnv(E::BilayerRightTensor, args...) = _pushleft(E, args...)

function _calITP_worker!(Ch::Channel, Ch_Timer::Channel, G::ImagTimeProxyGraph, ρ::MPO, Lock::ReentrantLock)

     TO = TimerOutput()
     @timeit TO "take" v = take!(Ch)

     S_clear = Int64[]
     if isa(v, Int64) # vertex
          st = get_prop(G.graph, v, :st)
          # load the environment tensor of current vertex
          if v == 1
               local E = BilayerLeftTensor{1,1}(isometry(codomain(ρ[1])[1], codomain(ρ[1])[1]))
          elseif v == 2
               local E = BilayerRightTensor{1,1}(isometry(domain(ρ[end])[end], domain(ρ[end])[end]))
          elseif any(x -> get_prop(G.graph, x, :st) == st, children(G, v))
               # otherwise, E is not used
               @timeit TO "loadEnv" local E = _load_Env(G, v)
          end

          @timeit TO "saveEnv" v ≤ 2 && _save_Env!(G, v, E)


          S_next = Union{Int64,NTuple{2,Int64}}[]
          for v_child in filter(x -> get_prop(G.graph, x, :st) == st, children(G, v))
               # push Env
               si = get_prop(G.graph, v_child, :si)
               O₁, O₂ = G.Ops[si][get_prop(G.graph, v_child, :idx_Op)]
               @timeit TO "pushEnv" local E_child = _pushEnv(E, ρ[si]', O₁, ρ[si], O₂)

               @timeit TO "saveEnv" _save_Env!(G, v_child, E_child)
               push!(S_next, v_child)
          end

          child_op = filter(children(G, v)) do child
               get_prop(G.graph, child, :st) != st
          end

          @timeit TO "lock" begin
               lock(Lock)
               try
                    set_prop!(G.graph, v, :passed, true)
                    for v_child in child_op
                         if get_prop(G.graph, v_child, :passed)
                              push!(S_next, st == :L ? (v, v_child) : (v_child, v))
                         end
                    end

                    # check if the current vertex can be cleared
                    if all(x -> get_prop(G.graph, st == :L ? Edge(v, x) : Edge(x, v), :passed), child_op)
                         push!(S_clear, v)
                    end

               catch
                    rethrow()
               finally
                    unlock(Lock)
               end
          end

          for s in S_next
               put!(Ch, s)
          end

     else # edge
          e = Edge(v...)
          @timeit TO "loadEnv" El = _load_Env(G, v[1])
          @timeit TO "loadEnv" Er = _load_Env(G, v[2])

          @timeit TO "trace" val = El * Er
          for ref in get_prop(G.graph, e, :Refs)
               ref[] = val
          end

          @timeit TO "lock" begin
               lock(Lock)
               try
                    set_prop!(G.graph, e, :passed, true)
                    # check if any vertex can be cleared
                    for x in filter(x -> _cond_clear(G, x; edge_only=true), v)
                         push!(S_clear, x)
                    end
               catch
                    rethrow()
               finally
                    unlock(Lock)
               end
          end

     end

     # clear
     @timeit TO "clear" for v in S_clear
          _clear_Env!(G, v)   
     end

     put!(Ch_Timer, TO)
     return nothing
end

function _load_Env(G::ImagTimeProxyGraph, v::Int64)
     return _load_Env(get_prop(G.graph, v, :E))
end
_load_Env(E::Union{BilayerLeftTensor,BilayerRightTensor}) = E
function _load_Env(filename::String)
     return deserialize(filename)
end
function _save_Env!(G::ImagTimeProxyGraph, v::Int64, E::Union{BilayerLeftTensor,BilayerRightTensor})
     filename = get_prop(G.graph, v, :E)
     if isnothing(filename)
          G.graph.vprops[v][:E] = E
     else
          serialize(filename, E)
     end
     return nothing
end

function _clear_Env!(G::ImagTimeProxyGraph, v::Int64)
     if isa(G.graph.vprops[v][:E], String)
          rm(G.graph.vprops[v][:E])
     end
     delete!(G.graph.vprops[v], :E)
     return nothing
end



