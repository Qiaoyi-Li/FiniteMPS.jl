mutable struct ImagTimeProxyGraph{L}
     Ops::Vector{Vector{NTuple{2,AbstractLocalOperator}}} # Ops[si][idx]
     Refs::Dict{String,Dict} # store the references of the values to be calculated
     graph::MetaDiGraph{Int64,Float64}
     function ImagTimeProxyGraph(L)

          Ops = Vector{Vector{NTuple{2,AbstractLocalOperator}}}(undef, L)
          for i in 1:L
               Ops[i] = Vector{NTuple{2,AbstractLocalOperator}}()
          end
          Refs = Dict{String,Dict}()
          graph = SimpleDiGraph(2) |> MetaDiGraph
          # left/right root vertex
          set_props!(graph, 1, Dict(:si => 0, :st => :L))
          set_props!(graph, 2, Dict(:si => L + 1, :st => :R))
          return new{L}(Ops, Refs, graph)
     end
end

function parent(G::ImagTimeProxyGraph{L}, v::Int64) where L
     get_prop(G.graph, v, :si) ∈ [0, L + 1] && return nothing
     if get_prop(G.graph, v, :st) == :L
          return inneighbors(G.graph, v)[1]
     elseif get_prop(G.graph, v, :st) == :R
          return outneighbors(G.graph, v)[1]
     else
          error("ambiguous parent for middle vertex!")
     end
end

function children(G::ImagTimeProxyGraph{L}, v::Int64) where L
     if get_prop(G.graph, v, :st) == :L
          return outneighbors(G.graph, v)
     elseif get_prop(G.graph, v, :st) == :R
          return inneighbors(G.graph, v)
     else
          error("ambiguous children for middle vertex!")
     end
end

function siblings(G::ImagTimeProxyGraph{L}, v::Int64) where L
     v_parent = parent(G, v)
     isnothing(v_parent) && return nothing

     return filter(children(G, v_parent)) do i
          i != v
     end
end

function rem_vertices!(g::MetaDiGraph{Int64}, vs::AbstractVector{Int64}; keep_order::Bool=false)
     # this is a naive implementation, which takes most of time in merge!
     # it may be optimized by directly modifying the keys via pointers however I am not familiar with the details of Dict in Julia 
     # hope MetaGraphs.jl can provide an official implementation in a future release

     vmap = rem_vertices!(g.graph, vs; keep_order=keep_order)
     vmap_inv = Dict(vmap[i] => i for i in 1:length(vmap))

     g.vprops = typeof(g.vprops)(i => g.vprops[vmap[i]] for i in 1:length(vmap))

     k_valid = filter(keys(g.eprops)) do k
          k.src ∉ vs && k.dst ∉ vs
     end
     g.eprops = typeof(g.eprops)(Edge(vmap_inv[k.src], vmap_inv[k.dst]) => g.eprops[k] for k in k_valid)

     return vmap
end

function merge!(G::ImagTimeProxyGraph{L}) where {L}

     v_rm = Set{Int64}()
     for si in 0:L
          @show si
          @time _left_merge!(G, v_rm, si)
          @time _right_merge!(G, v_rm, L + 1 - si)

          @time rem_vertices!(G.graph, collect(v_rm); keep_order =true)
          empty!(v_rm) 
     end

     _expand_left_tree!(G)

     return nothing
end

function _left_merge!(G::ImagTimeProxyGraph{L}, v_rm::Set{Int64}, si::Int64) where {L}


     lsv = filter(1:nv(G.graph)) do i
          get_prop(G.graph, i, :si) != si && return false
          get_prop(G.graph, i, :st) != :L && return false
          i ∈ v_rm && return false
          true
     end

     for v in lsv
          children = filter(outneighbors(G.graph, v)) do i
               get_prop(G.graph, i, :st) != :R
          end
          lsidx_Op = map(children) do i
               get_prop(G.graph, i, :idx_Op)
          end
          for j in 2:length(children)
               first_equal = findfirst(x -> x == lsidx_Op[j], lsidx_Op[1:j-1])
               isnothing(first_equal) && continue

               v_parent = children[first_equal]
               for v_next in outneighbors(G.graph, children[j])
                    if has_edge(G.graph, v_parent, v_next)
                         append!(get_prop(G.graph, Edge(v_parent, v_next), :Refs), get_prop(G.graph, Edge(children[j], v_next), :Refs))
                    else
                         add_edge!(G.graph, v_parent, v_next)
                         set_prop!(G.graph, Edge(v_parent, v_next), :Refs, get_prop(G.graph, Edge(children[j], v_next), :Refs))
                    end
               end

               # label children[j] as removed
               push!(v_rm, children[j])

               if get_prop(G.graph, children[first_equal], :st) != :L
                    set_prop!(G.graph, children[first_equal], :st, :L)
                    clear_props!(G.graph, Edge(v, children[first_equal]))
               end


          end

     end


     return nothing
end

function _right_merge!(G::ImagTimeProxyGraph{L}, v_rm::Set{Int64}, si::Int64) where {L}


     lsv = filter(1:nv(G.graph)) do i
          get_prop(G.graph, i, :si) != si && return false
          get_prop(G.graph, i, :st) != :R && return false
          i ∈ v_rm && return false
          true
     end

     for v in lsv
          children = filter(inneighbors(G.graph, v)) do i
               get_prop(G.graph, i, :st) != :L
          end
          lsidx_Op = map(children) do i
               get_prop(G.graph, i, :idx_Op)
          end
          for j in 2:length(children)
               first_equal = findfirst(x -> x == lsidx_Op[j], lsidx_Op[1:j-1])
               isnothing(first_equal) && continue

               v_parent = children[first_equal]
               for v_next in inneighbors(G.graph, children[j])
                    if has_edge(G.graph, v_next, v_parent)
                         append!(get_prop(G.graph, Edge(v_next, v_parent), :Refs), get_prop(G.graph, Edge(v_next, children[j]), :Refs))
                    else
                         add_edge!(G.graph, v_next, v_parent)
                         set_prop!(G.graph, Edge(v_next, v_parent), :Refs, get_prop(G.graph, Edge(v_next, children[j]), :Refs))
                    end
               end

               # label children[j] as removed
               push!(v_rm, children[j])

               if get_prop(G.graph, children[first_equal], :st) != :R
                    set_prop!(G.graph, children[first_equal], :st, :R)
                    clear_props!(G.graph, Edge(children[first_equal], v))
               end
          end
     end


     return nothing

end

function _expand_left_tree!(G::ImagTimeProxyGraph{L}) where {L}
     for si in 1:L
          lsv = filter(1:nv(G.graph)) do i
               get_prop(G.graph, i, :si) != si && return false
               return get_prop(G.graph, i, :st) == :M
          end
          for v in lsv
               v_in = inneighbors(G.graph, v)[1]
               clear_props!(G.graph, Edge(v_in, v))
               set_prop!(G.graph, v, :st, :L)
          end
     end
     return nothing
end








