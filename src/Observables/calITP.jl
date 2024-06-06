function calITP!(G::ImagTimeProxyGraph, ρ::MPO{L}; kwargs...) where {L}

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
     e_pass::Union{Nothing,Edge}=nothing)
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
          if verbose > 0 && iszero(num_count % showspacing)
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
               if _cond_clear(G, v)
                    delete!(G.graph.vprops[v], :E)
               end
          end
     end

     return Timer_acc
end

_pushEnv(E::BilayerLeftTensor, args...) = _pushright(E, args...)
_pushEnv(E::BilayerRightTensor, args...) = _pushleft(E, args...)

function _calITP_worker!(Ch::Channel, Ch_Timer::Channel, G::ImagTimeProxyGraph, ρ::MPO, Lock::ReentrantLock)

     TO = TimerOutput()
     @timeit TO "take" v = take!(Ch)

     if isa(v, Int64) # vertex
          si = get_prop(G.graph, v, :si)
          st = get_prop(G.graph, v, :st)
          if v == 1
               E = BilayerLeftTensor{1,1}(isometry(codomain(ρ[1])[1], codomain(ρ[1])[1]))
          elseif v == 2
               E = BilayerRightTensor{1,1}(isometry(domain(ρ[end])[end], domain(ρ[end])[end]))
          else
               O₁, O₂ = G.Ops[si][get_prop(G.graph, v, :idx_Op)]
               @timeit TO "pushEnv" E = _pushEnv(get_prop(G.graph, parent(G, v), :E),
                    ρ[si]', O₁, ρ[si], O₂)
          end

          # writing to the graph and checking according to the status may suffer data race
          S_next = Union{Int64,NTuple{2,Int64}}[]
          @timeit TO "reduce" begin
               lock(Lock)
               try
                    set_prop!(G.graph, v, :E, E)
                    set_prop!(G.graph, v, :passed, true)

                    # check if the parent vertex can be cleared
                    v_parent = parent(G, v)
                    @timeit TO "clear" begin
                         if v > 2 && _cond_clear(G, v_parent)
                              delete!(G.graph.vprops[v_parent], :E)
                         end
                    end

                    # add vertices and edges to be calculated into the channel
                    # only record them and put them after the lock 
                    for v_child in children(G, v)
                         if get_prop(G.graph, v_child, :st) == st
                              push!(S_next, v_child)
                         elseif get_prop(G.graph, v_child, :passed)
                              push!(S_next, st == :L ? (v, v_child) : (v_child, v))
                         end
                    end

                    # check if current vertex can be cleared
                    @timeit TO "clear" begin
                         if _cond_clear(G, v)
                              delete!(G.graph.vprops[v], :E)
                         end
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
          El = get_prop(G.graph, v[1], :E)
          Er = get_prop(G.graph, v[2], :E)
          @timeit TO "trace" val = El * Er
          for ref in get_prop(G.graph, e, :Refs)
               ref[] = val
          end

          @timeit TO "reduce" begin
               lock(Lock)
               try
                    set_prop!(G.graph, e, :passed, true)
                    # check if any vertex can be cleared
                    @timeit TO "clear" begin
                         for x in filter(x -> _cond_clear(G, x), v)
                              delete!(G.graph.vprops[x], :E)
                         end
                    end
               catch
                    rethrow()
               finally
                    unlock(Lock)
               end
          end

     end

     put!(Ch_Timer, TO)
     return nothing
end

