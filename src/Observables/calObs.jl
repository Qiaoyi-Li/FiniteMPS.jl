"""
     calObs!(Tree::ObservableTree, Ψ::AbstractMPS{L}; kwargs...) -> Tree::InteractionTree

Calculate observables respect to state `Ψ`, the info to tell which observables to calculate is stored in `Tree`. The results are stored in each leave node of `Tree`. 

# Kwargs
     verbose::Int64 == 0
Print the `TimerOutput` of each site if `verbose > 0`.

     GCstep::Bool == false
Call `manualGC()` after each site if `true`.
"""
function calObs!(Tree::ObservableTree, Ψ::AbstractMPS{L}; kwargs...) where {L}
     # default El
     for node in Tree.Root.children
          node.value[1] = isometry(codomain(Ψ[1])[1], codomain(Ψ[1])[1])
     end

     calObs!(Tree.Root, Ψ; kwargs...)
     manualGC()

     return Tree
end

function calObs!(Root::InteractionTreeNode, Ψ::AbstractMPS{L}; kwargs...) where {L}
     @assert isroot(Root) && treeheight(Root) == L + 1
     @assert Center(Ψ)[2] == 1 # note right-canonical form is used

     si = 0
     # generate a stack to store the nodes with the same si
     stack = Vector{InteractionTreeNode}(undef, 0)
     stack_pre = Vector{InteractionTreeNode}(undef, 0)
     for node in StatelessBFS(Root)
          isnothing(node.Op) && continue
          if node.Op.si == 0
               # initialize stack_pre
               push!(stack_pre, node)
               continue
          end
          if node.Op.si > si
               if si > 0
                    # calculate the El respect to the nodes in stack
                    _calObs_stack!(stack, Ψ[si]', Ψ[si]; kwargs...)
                    # free
                    while !isempty(stack_pre)
                         node_pop = pop!(stack_pre)
                         node_pop.value = nothing
                    end

                    while !isempty(stack)
                         push!(stack_pre, pop!(stack))
                    end

               end

               si = node.Op.si
          end

          si == node.Op.si && push!(stack, node)
     end

     # leave nodes
     _calObs_stack!(stack, Ψ[L]', Ψ[L]; kwargs...)

     return nothing

end

function _calObs_stack!(stack::Vector{InteractionTreeNode}, A::AdjointMPSTensor, B::MPSTensor; kwargs...)

     GCstep::Bool = get(kwargs, :GCstep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     LocalTimer = TimerOutput()


     if get_num_workers() > 1

          tasks = Vector{Future}(undef, length(stack))
          El = parent(stack[1]).value[1]
          @timeit LocalTimer "calObs_stack!" begin
               for i in 1:length(stack)
                    if i > 1 && parent(stack[i-1]) != parent(stack[i])
                         El = parent(stack[i]).value[1]
                    end
                    @timeit LocalTimer "spawn_task" begin
                         let El = El, Op = stack[i].Op
                              tasks[i] = @spawnat :any _update_node(Op, El, A, B)
                         end
                    end

               end
          end

          @timeit LocalTimer "fetch_El" begin
               lsTimer = Vector{TimerOutput}(undef, length(stack))
               @threads for i in 1:length(stack)
                    stack[i].value[1], stack[i].Op.strength, lsTimer[i] = fetch(tasks[i])
               end
          end
          merge!(LocalTimer, lsTimer...; tree_point=["calObs_stack!"])

     else

          @timeit LocalTimer "calObs_stack!" begin
               @floop GlobalThreadsExecutor for node in stack
                    tmp = let El = parent(node).value[1]
                         _update_node!(node, El, A, B)
                    end  
                    @reduce() do (Timer_acc = TimerOutput(); tmp)
                         Timer_acc = merge!(Timer_acc, tmp)
                    end
               end
          end
          merge!(LocalTimer, Timer_acc; tree_point=["calObs_stack!"])

     end

     GCstep && manualGC(LocalTimer)

     if verbose > 0
          show(LocalTimer; title="site $(stack[1].Op.si)")
          println()
          flush(stdout)
     end

     return nothing

end

function _update_node(Op::AbstractLocalOperator, El::LocalLeftTensor, A::AdjointMPSTensor, B::MPSTensor)
     # return updated El and strength

     LocalTimer = TimerOutput()

     if isnan(Op.strength)
          # propagate
          Op.strength = 1
          @timeit LocalTimer "_pushright" El_new = _pushright(El, A, Op, B)
          return El_new, NaN, LocalTimer
     else
          fac = Op.strength
          @assert abs(fac) == 1 # just convention, -1 for some fermionic cases
          Op.strength = 1 # we do not want to propagate -El !!
          @timeit LocalTimer "_pushright" El_new = _pushright(El, A, Op, B)
          @timeit LocalTimer "trace" strength = fac * tr(El_new.A) # add fac(±1) here
          return El_new, strength, LocalTimer
     end

end

function _update_node!(node::InteractionTreeNode, El::LocalLeftTensor, A::AdjointMPSTensor, B::MPSTensor)
     # directly update El

     LocalTimer = TimerOutput()

     if isnan(node.Op.strength)
          # propagate
          node.Op.strength = 1
          @timeit LocalTimer "_pushright" node.value[1] = _pushright(El, A, node.Op, B)
          node.Op.strength = NaN

     else
          fac = node.Op.strength
          @assert abs(fac) == 1 # just convention, -1 for some fermionic cases
          node.Op.strength = 1 # we do not want to propagate -El !!
          @timeit LocalTimer "_pushright" El_new = _pushright(El, A, node.Op, B)
          @timeit LocalTimer "trace" node.Op.strength = fac * tr(El_new.A) # add fac(±1) here
          node.value[1] = El_new

     end

     return LocalTimer

end
