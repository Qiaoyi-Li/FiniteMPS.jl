function convert(::Type{Dict}, Root::InteractionTreeNode, name::NTuple{N,String}; kwargs...) where {N}

     Rslt = Dict{NTuple{N,Int64},Number}() # Tuple si => value

     ch = Vector{String}(undef, treeheight(Root))
     for node in PreOrderDFS(Root)
          isnothing(node.Op) && continue
          (si = node.Op.si) == 0 && continue
          ch[si] = getOpName(node.Op)
          if !isnan(node.Op.strength)
               si_tuple = _parseOp(view(ch, 1:si), name; kwargs...)
               !isnothing(si_tuple) && (Rslt[si_tuple] = node.Op.strength)
          end
     end

     return Rslt
end
function convert(::Type{T}, Root::InteractionTreeNode, name::Vector;kwargs...) where T
     name = broadcast(x -> String.(x), name)
     Rslt = map(name) do x
          convert(T, Root, x; kwargs...)
     end
     return NamedTuple{Tuple(Symbol.(prod.(name)))}(Rslt)

end
function convert(::Type{T}, Root::InteractionTreeNode, name::NTuple{N,Symbol};kwargs...) where {T, N}
     return convert(T, Root, String.(name); kwargs...) 
end
function convert(::Type{T}, Tree::ObservableTree, name::Any; kwargs...) where T
     return convert(T, Tree.Root, name; kwargs...) 
end



function _parseOp(ch::AbstractVector{String}, name::NTuple{N,String}; fuzzyZ=true) where {N}
     # return si(length N Tuple) or nothing(mismatch)

     # check length
     name_I = getOpName(IdentityOperator(0))
     (mapreduce(x -> x != name_I && x != "Z", +, ch) != N) && return nothing


     si = Vector{Int64}(undef, N)
     if fuzzyZ
          ch = map(ch) do str
               str[1] == 'Z' ? str[2:end] : str
          end
     end

     valid = ones(Bool, N)
     for i = 1:length(ch)
          idx = findfirst(x -> valid[x] && name[x] == ch[i], 1:N)
          isnothing(idx) && continue
          valid[idx] = false
          si[idx] = i
     end

     # TODO deal with cases e.g. (1, 2, 2, 3)

     return any(valid) ? nothing : Tuple(si)
end