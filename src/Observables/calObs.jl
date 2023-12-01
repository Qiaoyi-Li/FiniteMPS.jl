"""
     calObs!(Tree::ObservableTree, Ψ::AbstractMPS{L}; kwargs...) -> Tree::InteractionTree

Calculate observables respect to state `Ψ`, the info to tell which observables to calculate is stored in `Tree`. The results are stored in each leave node of `Tree`. 

Note only multi-threading backend is supported now.
"""
function calObs!(Tree::ObservableTree, Ψ::AbstractMPS{L}; kwargs...) where {L}

     if get_num_workers() > 1
          # TODO
          return _calObs_processing!(Tree, Ψ; kwargs...)
     elseif get_num_threads_julia() > 1
          if get(kwargs, :disk, false)
               return _calObs_threading!(Tree, Ψ, StoreDisk(); kwargs...)
          else
               return _calObs_threading!(Tree, Ψ, StoreMemory(); kwargs...)
          end
     else
          # TODO
          return _calObs_serial!(Tree, Ψ; kwargs...)
     end
end

function _calObs_threading!(Tree::ObservableTree, Ψ::AbstractMPS{L}, ::StoreMemory; kwargs...) where {L}

     ntasks::Int64 = get(kwargs, :ntasks, get_num_threads_julia() - 1)
     GCspacing::Int64 = get(kwargs, :GCspacing, 100)
     verbose::Int64 = get(kwargs, :verbose, 0)
     showtimes::Int64 = get(kwargs, :showtimes, 100)

     Ch = Channel{Tuple{InteractionTreeNode,LocalLeftTensor}}(Inf)
     Ch_Timer = Channel{TimerOutput}(Inf)

     # workers
     map(1:ntasks) do _
          Threads.@spawn while isopen(Ch)
               to = _calObs_worker!(Ch, Ψ)
               put!(Ch_Timer, to)
          end
     end

     # initialize the recursion
     El::LocalLeftTensor = isometry(codomain(Ψ[1])[1], codomain(Ψ[1])[1])
     for node in Tree.Root.children
          put!(Ch, (node, El))
     end

     # main thread
     Timer_acc = TimerOutput()
     num_tot = 0
     for _ in PreOrderDFS(Tree.Root)
          num_tot += 1
     end
     num_count = 0
     showspacing::Int64 = cld(num_tot, showtimes)
     while num_count < num_tot - 1
          to = take!(Ch_Timer)
          merge!(Timer_acc, to)
          num_count += 1
          if num_count % GCspacing == 0
               manualGC(Timer_acc)
          end
          if verbose > 0 && num_count % showspacing == 0
               show(Timer_acc; title="$(num_count) / $(num_tot)")
               println()
               flush(stdout)
          end
     end

     return Timer_acc
end

# function _calObs_threading!(Tree::ObservableTree, Ψ::AbstractMPS{L}, ::StoreDisk; kwargs...) where {L}
#      # TODO to be optimized

#      ntasks::Int64 = get(kwargs, :ntasks, get_num_threads_julia() - 1)
#      GCspacing::Int64 = get(kwargs, :GCspacing, 100)
#      verbose::Int64 = get(kwargs, :verbose, 0)
#      showtimes::Int64 = get(kwargs, :showtimes, 100)
#      cachesize::Int64 = get(kwargs, :cachesize, 4 * ntasks)
#      sleeptime::Float64 = get(kwargs, :sleeptime, 1)


#      Dir = mktempdir()
#      Ch = Channel{Tuple{InteractionTreeNode,LocalLeftTensor}}(cachesize)
#      Ch_swap = Channel{String}(Inf)
#      Ch_Timer = Channel{TimerOutput}(Inf)

#      # swap control
#      task_swap = Threads.@spawn while isopen(Ch)
#           sz = Base.n_avail(Ch)
#           if sz < ntasks && isready(Ch_swap)
#                str = take!(Ch_swap)
#                put!(Ch, deserialize(str))
#                # clean 
#                rm(str; force=true)
#           elseif sz > cachesize - ntasks
#                # save to disk
#                str = tempname(Dir) * ".bin"
#                serialize(str, take!(Ch))
#                put!(Ch_swap, str)
#           else
#                sleep(sleeptime)
#           end
#      end

#      # workers
#      map(1:ntasks) do _
#           Threads.@spawn while isopen(Ch)
#                to = _calObs_worker!(Ch, Ψ)
#                put!(Ch_Timer, to)
#           end
#      end

#      # initialize the recursion
#      El::LocalLeftTensor = isometry(codomain(Ψ[1])[1], codomain(Ψ[1])[1])
#      for node in Tree.Root.children
#           put!(Ch, (node, El))
#      end

#      # main thread
#      Timer_acc = TimerOutput()
#      num_tot = 0
#      for _ in PreOrderDFS(Tree.Root)
#           num_tot += 1
#      end
#      num_count = 0
#      showspacing::Int64 = cld(num_tot, showtimes)
#      while num_count < num_tot - 1
#           to = take!(Ch_Timer)
#           merge!(Timer_acc, to)
#           num_count += 1
#           if num_count % GCspacing == 0
#                manualGC(Timer_acc)
#           end
#           if verbose > 0 && num_count % showspacing == 0
#                show(Timer_acc; title="$(num_count) / $(num_tot)")
#                println()
#                flush(stdout)
#           end
#      end

#      return Timer_acc

# end

function _calObs_worker!(Ch::Channel, Ψ::AbstractMPS{L}) where {L}
     LocalTimer = TimerOutput()

     @timeit LocalTimer "take" (node_parent, El) = take!(Ch)

     si = node_parent.Op.si + 1
     si == L + 1 && return LocalTimer

     let A = Ψ[si]', B = Ψ[si]
          for node in node_parent.children
               if isnan(node.Op.strength)
                    # propagate
                    node.Op.strength = 1
                    @timeit LocalTimer "pushright" El_next = _pushright(El, A, node.Op, B)
                    node.Op.strength = NaN

               else
                    fac = node.Op.strength
                    @assert abs(fac) == 1 # just convention, -1 for some fermionic cases
                    node.Op.strength = 1 # we do not want to propagate -El !!
                    @timeit LocalTimer "pushright" El_next = _pushright(El, A, node.Op, B)
                    @timeit LocalTimer "trace" node.Op.strength = fac * tr(El_next.A) # add fac(±1) here
               end
               # print(node.Op, " ")
               @timeit LocalTimer "put" put!(Ch, (node, El_next))
          end
     end
     return LocalTimer
end
