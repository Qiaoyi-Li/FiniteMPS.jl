function calObs!(Tree::ObservableTree{L}, Ψ::AbstractMPS{L}; kwargs...) where L
     
     # make sure the tree is merged
     merge!(Tree)
     
     disk::Bool = get(kwargs, :disk, false)

     # generate a map from nodes to [status, filename]
     nodemap = Dict{InteractionTreeNode, Dict{Symbol, Any}}()
     if disk 
          dir =  mktempdir()
     end
     i = 0
     for R in [Tree.RootL, Tree.RootR], node in PreOrderDFS(R)
          si = node.Op[1]
          # i is unique for each node
          filename = disk ? joinpath(dir, "node_$(si)_$(i).bin") : ""
          nodemap[node] = Dict{Symbol, Any}(:st => false, :filename => filename)
          i += 1
     end

     # two caches
     maxsize = disk ? get(kwargs, :maxsize, 100) : i  
     # finalizer to store in disk
     function f(k, v)
          filename = nodemap[k][:filename]
          !isempty(filename) && serialize(filename, v)
          # TODO test Ptr array 
          return nothing 
     end
     cacheL = LRU{InteractionTreeNode, LocalLeftTensor}(maxsize = maxsize, finalizer = f)
     cacheR = LRU{InteractionTreeNode, LocalRightTensor}(maxsize = maxsize, finalizer = f)

     # initialize boundary environment tensors
     El::LocalLeftTensor = isometry(codomain(Ψ[1])[1], codomain(Ψ[1])[1])
     Er::LocalRightTensor = isometry(domain(Ψ[end])[end], domain(Ψ[end])[end])
     cacheL[Tree.RootL] = El
     nodemap[Tree.RootL][:st] = true
     cacheR[Tree.RootR] = Er
     nodemap[Tree.RootR][:st] = true
     
     if get(kwargs, :serial, false) || get_num_threads_julia() == 1
          _calObs_serial!(Tree, Ψ, nodemap, cacheL, cacheR; kwargs...)
     else
          @assert false # TODO
     end

     # cleanup 
     for c in [cacheL, cacheR]
          # empty filename to avoid unnecessary serialization
          for (k, v) in c
               nodemap[k][:filename] = ""
          end
          empty!(c)
     end
     if disk
          rm(dir; recursive = true, force = true)
     end
     
     return nothing
end

function _calObs_serial!(Tree::ObservableTree{L},
     Ψ::AbstractMPS{L},
     nodemap::Dict{InteractionTreeNode, Dict{Symbol, Any}}, 
     cacheL::LRU{InteractionTreeNode, LocalLeftTensor},
     cacheR::LRU{InteractionTreeNode, LocalRightTensor};
     kwargs...) where L 

     # finish right tree first 
     for node in StatelessBFS(Tree.RootR)
          si = node.Op[1] - 1
          # get Er 
          if haskey(cacheR, node)
               Er = cacheR[node] 
          else
               # load from disk and store in cache
               Er = deserialize(nodemap[node][:filename])
               cacheR[node] = Er
          end

          # process all children
          for ch in node.children
               Op = deepcopy(Tree.Ops[si][ch.Op[2]])
               Op.strength[] = 1.0
               cacheR[ch] = _pushleft(Er, Ψ[si]', Op, Ψ[si])
               nodemap[ch][:st] = true
          end
     end

     # left tree
     for node in StatelessBFS(Tree.RootL)
          si = node.Op[1] + 1
          # get El
          if haskey(cacheL, node)
               El = cacheL[node]
          else
               El = deserialize(nodemap[node][:filename])
               cacheL[node] = El
          end

          # process all children
          for ch in node.children
               Op = deepcopy(Tree.Ops[si][ch.Op[2]])
               Op.strength[] = 1.0
               cacheL[ch] = _pushright(El, Ψ[si]', Op, Ψ[si])
               nodemap[ch][:st] = true
          end

          # process all Intrs 
          # deal with duplicate channels
          node2val = Dict{InteractionTreeNode, Number}()
          for ch in node.Intrs
               node_R = ch.LeafR
               if haskey(node2val, node_R)
                    ch.ref[] = node2val[node_R]
                    continue 
               end  
               if haskey(cacheR, node_R)
                    Er = cacheR[node_R]
               else
                    Er = deserialize(nodemap[node_R][:filename])
                    cacheR[node_R] = Er
               end
               node2val[node_R] = ch.ref[] = El * Er
          end
               
          # El is no longer needed
          rm(nodemap[node][:filename]; force = true)
          nodemap[node][:filename] = ""
          delete!(cacheL, node)
     end
          
     for (k,v) in nodemap
          @assert v[:st] 
     end

     return nothing
end


# """
#      calObs!(Tree::ObservableTree, Ψ::AbstractMPS{L}; kwargs...) -> Timer::TimerOutput

# Calculate observables respect to state `Ψ`, the info to tell which observables to calculate is stored in `Tree`. The results are stored in each leaf node of `Tree`. Note the value in each node will be in-place updated, so do not call this function twice with the same `Tree` object.

# # Kwargs
#      serial::Bool = false
# Force to compute in serial mode, usually used for debugging.
# """
# function calObs!(Tree::ObservableTree, Ψ::AbstractMPS{L}; kwargs...) where {L}
#      @assert Center(Ψ) == [1,1] "MPS must be canonicalized at the first site for calObs! to work correctly!"

#      if get(kwargs, :serial, false)
#           return _calObs_serial!(Tree, Ψ; kwargs...)
#      end

#      if get_num_workers() > 1
#           # TODO
#           return _calObs_processing!(Tree, Ψ; kwargs...)
#      elseif get_num_threads_julia() > 1
#           if get(kwargs, :disk, false)
#                return _calObs_threading!(Tree, Ψ, StoreDisk(); kwargs...)
#           else
#                return _calObs_threading!(Tree, Ψ, StoreMemory(); kwargs...)
#           end
#      else
#           # fallback to serial
#           return _calObs_serial!(Tree, Ψ; kwargs...)
#      end
# end

# function _calObs_serial!(Tree::ObservableTree, Ψ::AbstractMPS{L}; kwargs...) where {L}

#      GCspacing::Int64 = get(kwargs, :GCspacing, 100)
#      verbose::Int64 = get(kwargs, :verbose, 0)
#      showtimes::Int64 = get(kwargs, :showtimes, 100)

#      # store the map from nodes to corresponding left environment tensors
#      Dict_El = Dict{InteractionTreeNode, LocalLeftTensor}()

#      # print
#      Timer_acc = TimerOutput()
#      num_tot = 0
#      for _ in PreOrderDFS(Tree.Root)
#           num_tot += 1
#      end
#      num_count = 0
#      showspacing::Int64 = cld(num_tot, showtimes)
#      GC_count = 0

#      for node in PreOrderDFS(Tree.Root)
#           num_count += 1
#           isnothing(node.Op) && continue
#           si = node.Op.si
#           if si == 0 # initialize
#                Dict_El[node] = isometry(codomain(Ψ[1])[1], codomain(Ψ[1])[1])
#                continue
#           end

#           # update
#           El = _update_node!(node.Op, Dict_El[node.parent], Ψ[si]', Ψ[si], Timer_acc)

#           # store El only if there exist children
#           if !isempty(node.children)
#                Dict_El[node] = El
#           end

#           # delete entry if all children are visited
#           if node === last(node.parent.children)
#                delete!(Dict_El, node.parent)
#           end

#           # print
#           if verbose > 0 && iszero(num_count % showspacing)
#                show(Timer_acc; title="$(num_count) / $(num_tot)")
#                println()
#                flush(stdout)
#           end

#           # manual GC
#           GC_count += 1
#           if GC_count == GCspacing
#                GC_count = 0
#                manualGC(Timer_acc)
#           end

#      end

#      return Timer_acc
# end

# function _calObs_threading!(Tree::ObservableTree, Ψ::AbstractMPS{L}, ::StoreMemory; kwargs...) where {L}

#      ntasks::Int64 = get(kwargs, :ntasks, get_num_threads_julia() - 1)
#      @assert ntasks ≤ get_num_threads_julia() - 1 # avoid conflict
#      GCspacing::Int64 = get(kwargs, :GCspacing, 100)
#      verbose::Int64 = get(kwargs, :verbose, 0)
#      showtimes::Int64 = get(kwargs, :showtimes, 100)
#      cachesize::Int64 = get(kwargs, :cachesize, 4 * ntasks)

#      # Ch = Channel{Tuple{InteractionTreeNode,LocalLeftTensor}}(Inf)
#      Ch = Channel{Tuple{InteractionTreeNode,LocalLeftTensor}}(cachesize)
#      Ch_swap = Channel{Tuple{InteractionTreeNode,LocalLeftTensor}}(Inf)
#      Ch_Timer = Channel{Tuple{Int64, TimerOutput}}(Inf) # (num_count, TimerOutput)

#      # control swap
#      task_swap = Threads.@spawn while isopen(Ch)
#           sz = Base.n_avail(Ch)
#           if sz < div(cachesize, 2) && isready(Ch_swap)
#                put!(Ch, take!(Ch_swap))
#           else
#                sleep(0.01)
#           end
#      end

#      # workers
#      task_c = map(1:ntasks) do _
#           Threads.@spawn while isopen(Ch)
#                to = _calObs_worker!(Ch, Ch_swap, Ψ)
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
#      num_leaves = 0
#      for _ in Leaves(Tree.Root)
#           num_leaves += 1
#      end
#      num_count = 0
#      showspacing::Int64 = cld(num_leaves, showtimes)
#      GC_count = 0
#      num_left = num_leaves
#      while num_count < num_left
#           # check and rethrow if any task failed
#           for task in task_c
#                istaskfailed(task) && fetch(task)
#           end
#           (num, to) = take!(Ch_Timer)
#           merge!(Timer_acc, to)
#           num_count += num
#           GC_count += 1
#           if GC_count == GCspacing
#                GC_count = 0
#                manualGC(Timer_acc)
#           end
#           if verbose > 0 && num_count ≥ showspacing
#                num_left -= num_count
#                num_count = 0
#                show(Timer_acc; title="$(num_leaves - num_left) / $(num_leaves)")
#                println()
#                flush(stdout)
#           end
#      end

#      # kill tasks
#      close(Ch)
#      close(Ch_swap)
#      close(Ch_Timer)

#      return Timer_acc
# end

# function _calObs_worker!(Ch::Channel, Ch_swap::Channel, Ψ::AbstractMPS{L}) where {L}
#      LocalTimer = TimerOutput()

#      worklist = Vector{Tuple{InteractionTreeNode,LocalLeftTensor}}(undef, 0)
#      leaves_count = 0
#      @timeit LocalTimer "take" push!(worklist, take!(Ch))
#      while !isempty(worklist)
#           (node_parent, El) = pop!(worklist)
#           si = node_parent.Op.si + 1
#           si > L && return LocalTimer

#           let A = Ψ[si]', B = Ψ[si]
#                for node in node_parent.children
#                     El_child = _update_node!(node.Op, El, A, B, LocalTimer)
#                     if isempty(node.children) # leaf node
#                          leaves_count += 1
#                          continue
#                     end

#                     if length(worklist) < 1 # choose as next parent
#                          push!(worklist, (node, El_child))
#                     else # put to swap
#                          @timeit LocalTimer "put" put!(Ch_swap, (node, El_child))
#                     end
#                end
#           end

#      end
#      return leaves_count, LocalTimer
# end

# function _update_node!(Op::AbstractLocalOperator, El::LocalLeftTensor, A::AdjointMPSTensor, B::MPSTensor, LocalTimer::TimerOutput)

#      if isnan(Op.strength)
#           # propagate
#           Op.strength = 1
#           @timeit LocalTimer "pushright" El_next = _pushright(El, A, Op, B)
#           Op.strength = NaN
#      else
#           fac = Op.strength
#           @assert abs(fac) == 1 # just convention, -1 for some fermionic cases
#           Op.strength = 1 # we do not want to propagate -El !!
#           @timeit LocalTimer "pushright" El_next = _pushright(El, A, Op, B)
#           @timeit LocalTimer "trace" Op.strength = fac * tr(El_next.A) # add fac(±1) here
#      end
#      return El_next
# end
