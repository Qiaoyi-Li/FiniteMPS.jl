function addITP2!(G::ImagTimeProxyGraph{L},
     Op::NTuple{2,AbstractTensorMap},
     si::NTuple{2,Int64},
     Opname::NTuple{2,Union{Symbol,String}},
     ValueType::Type{<:Number}=Number;
     ITPname::Union{Symbol,String}=prod(string.(Opname)),
     Z::Union{Nothing,AbstractTensorMap}=nothing,
) where {L}

     @assert all(x -> 1 ≤ x ≤ L, si)

     if !haskey(G.Refs, ITPname)
          G.Refs[ITPname] = Dict{NTuple{2,Int64},Ref{ValueType}}()
     end
     G.Refs[ITPname][si] = Ref{ValueType}()

     O₁ = OnSiteInteractionIterator{L}(_rightOp(LocalOperator(Op[1], Opname[1], si[1])), Z)
     O₂ = OnSiteInteractionIterator{L}(_rightOp(LocalOperator(Op[2], Opname[2], si[2])), Z)
     return _addITP!(G, O₁, O₂, G.Refs[ITPname][si])

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
