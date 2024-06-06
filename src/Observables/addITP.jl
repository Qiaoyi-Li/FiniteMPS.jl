function addITP2!(G::ImagTimeProxyGraph{L},
     Op::NTuple{2,AbstractTensorMap},
     si::NTuple{2,Int64},
     Opname::NTuple{2,Union{Symbol,String}},
     ValueType::Type{<:Number}=Number;
     ITPname::Union{Symbol,String}=prod(string.(Opname)),
     Z::Union{Nothing,AbstractTensorMap}=nothing,
     merge::Bool=true
) where {L}

     @assert all(x -> 1 ≤ x ≤ L, si)

     if !haskey(G.Refs, ITPname)
          G.Refs[ITPname] = Dict{NTuple{2,Int64},Ref{ValueType}}()
     end
     G.Refs[ITPname][si] = Ref{ValueType}()

     O₁ = OnSiteInteractionIterator{L}(_rightOp(LocalOperator(Op[1], Opname[1], si[1])), Z)
     O₂ = OnSiteInteractionIterator{L}(_rightOp(LocalOperator(Op[2], Opname[2], si[2])), Z)
     if merge
          return _addITP_merge!(G, O₁, O₂, G.Refs[ITPname][si])
     else
          return _addITP!(G, O₁, O₂, G.Refs[ITPname][si])
     end
end

function _addITP!(G::ImagTimeProxyGraph{L},
     O₁::AbstractInteractionIterator{L},
     O₂::AbstractInteractionIterator{L},
     ref::Ref) where {L}

     idx_v = nv(G.graph)

     for (si, A, B) in zip(1:L, O₁, O₂)
          add_vertex!(G.graph)
          idx_v += 1

          idx_Op = findfirst(x -> x == (A, B), G.Ops[si])
          if isnothing(idx_Op)
               push!(G.Ops[si], (A, B))
               idx_Op = length(G.Ops[si])
          end

          set_props!(G.graph, idx_v, Dict(:si => si, :st => :M, :idx_Op => idx_Op))

          idx_left = si == 1 ? 1 : idx_v - 1
          add_edge!(G.graph, idx_left, idx_v)
          set_prop!(G.graph, Edge(idx_left, idx_v), :Refs, [ref,])
     end

     # edge to right root
     add_edge!(G.graph, idx_v, 2)
     set_prop!(G.graph, Edge(idx_v, 2), :Refs, [ref,])

     return nothing
end

function _addITP_merge!(G::ImagTimeProxyGraph{L},
     O₁::AbstractInteractionIterator{L},
     O₂::AbstractInteractionIterator{L},
     ref::Ref) where {L}

     num_v = nv(G.graph)
     lsO₁ = collect(O₁)
     lsO₂ = collect(O₂)

     # left to right
     v_last = 1
     for si in 1:div(L, 2)
          A = lsO₁[si]
          B = lsO₂[si]

          idx_Op = findfirst(x -> x == (A, B), G.Ops[si])
          # try to find the existed vertex
          if isnothing(idx_Op)
               new_flag = true
          else
               lsv = outneighbors(G.graph, v_last)
               idx_v = findfirst(lsv) do i
                    # compare idx instead of the operator to accelerate
                    return idx_Op == get_prop(G.graph, i, :idx_Op)
               end
               new_flag = isnothing(idx_v)
          end

          if new_flag
               # add a new one
               add_vertex!(G.graph)
               num_v += 1
               if isnothing(idx_Op)
                    push!(G.Ops[si], (A, B))
                    idx_Op = length(G.Ops[si])
               end
               set_props!(G.graph, num_v, Dict(:si => si, :st => :M, :idx_Op => idx_Op))
               add_edge!(G.graph, v_last, num_v)
               set_prop!(G.graph, Edge(v_last, num_v), :Refs, [ref,])

               # check if the parent is a left vertex now 
               if length(outneighbors(G.graph, v_last)) > 1 && get_prop(G.graph, v_last, :st) != :L
                    set_prop!(G.graph, v_last, :st, :L)
                    # clear refs
                    clear_props!(G.graph, Edge(inneighbors(G.graph, v_last)[1], v_last))
               end
               v_last = num_v
          else
               v_last = lsv[idx_v]
          end
     end

     # right to left
     v_right = 2
     for si in reverse(div(L, 2)+1:L)
          A = lsO₁[si]
          B = lsO₂[si]

          idx_Op = findfirst(x -> x == (A, B), G.Ops[si])
          # try to find the existed vertex
          if isnothing(idx_Op)
               new_flag = true
          else
               lsv = inneighbors(G.graph, v_right)
               idx_v = findfirst(inneighbors(G.graph, v_right)) do i
                    # compare idx instead of the operator to accelerate
                    return idx_Op == get_prop(G.graph, i, :idx_Op)
               end
               new_flag = isnothing(idx_v)
          end

          if new_flag
               # add a new one
               add_vertex!(G.graph)
               num_v += 1
               if isnothing(idx_Op)
                    push!(G.Ops[si], (A, B))
                    idx_Op = length(G.Ops[si])
               end
               set_props!(G.graph, num_v, Dict(:si => si, :st => :M, :idx_Op => idx_Op))
               add_edge!(G.graph, num_v, v_right)
               set_prop!(G.graph, Edge(num_v, v_right), :Refs, [ref,])

               # check if the parent is a right vertex now 
               if length(inneighbors(G.graph, v_right)) > 1 && get_prop(G.graph, v_right, :st) != :R
                    set_prop!(G.graph, v_right, :st, :R)
                    # clear refs
                    clear_props!(G.graph, Edge(v_right, outneighbors(G.graph, v_right)[1]))
               end
               v_right = num_v
          else
               v_right = lsv[idx_v]
          end
     end

     # middle edge
     add_edge!(G.graph, v_last, v_right)
     set_prop!(G.graph, Edge(v_last, v_right), :Refs, [ref,])
     # check if the middle two vertices are left and right
     if get_prop(G.graph, v_last, :st) != :L && length(outneighbors(G.graph, v_last)) > 1
          set_prop!(G.graph, v_last, :st, :L)
          clear_props!(G.graph, Edge(inneighbors(G.graph, v_last)[1], v_last))
     end
     if get_prop(G.graph, v_right, :st) != :R && length(inneighbors(G.graph, v_right)) > 1
          set_prop!(G.graph, v_right, :st, :R)
          clear_props!(G.graph, Edge(v_right, outneighbors(G.graph, v_right)[1]))
     end

     return nothing
end
