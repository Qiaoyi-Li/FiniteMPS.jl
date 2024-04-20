function convert(::Type{Dict}, Root::InteractionTreeNode; kwargs...) 

     T = Dict{Tuple{Vararg{Int64}}, Number}
     Rslt = Dict{String, T}() # Tuple si => value

     for node in PreOrderDFS(Root)
          isnothing(node.Op) && continue
          (si = node.Op.si) == 0 && continue
          if !isnan(node.Op.strength)
               name = node.value[1]
               sites = Tuple(node.value[2:end])
               if !haskey(Rslt, name)
                    Rslt[name] = T()
               end
               Rslt[node.value[1]][sites] = node.Op.strength 
          end
     end

     return Rslt
end
function convert(::Type{NamedTuple}, Root::InteractionTreeNode; kwargs...) 

     Rslt = convert(Dict, Root; kwargs...)
     k = keys(Rslt) .|> Symbol |> Tuple
     return NamedTuple{k}(values(Rslt))
end

function convert(::Type{T}, Tree::ObservableTree; kwargs...) where T
     return convert(T, Tree.Root; kwargs...) 
end
