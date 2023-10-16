"""
     canonicalize!(obj::AbstractEnvironment, 
          siL::Int64
          [, siR::Int64 = siL]; kwargs...) -> obj::AbstractEnvironment

Canonicalize the environment s.t. at least `El[i ≤ siL]` and `Er[i ≥ siR]` are valid.

# Kwargs
     free::Bool = true
If `true`, call `free!(obj)` to free the local environment tensors which are no longer required. Details see `free!`.
"""
function canonicalize!(obj::AbstractEnvironment{L}, siL::Int64, siR::Int64; kwargs...) where {L}
     free::Bool = get(kwargs, :free, true)

     @assert siL ≥ 1 && siR ≤ L
     # pushleft/right if needed

     while obj.Center[1] < siL
          pushright!(obj)
     end

     while obj.Center[2] > siR
          pushleft!(obj)
     end

     if free
          obj.Center[:] = [siL, siR]
          free!(obj)
     end
     return obj
end
canonicalize!(obj::AbstractEnvironment, si::Int64; kwargs...) = canonicalize!(obj, si, si; kwargs...)
function canonicalize!(obj::AbstractEnvironment, si::Vector{Int64}; kwargs...)
     @assert length(si) == 2
     return canonicalize!(obj, si[1], si[2]; kwargs...)
end

"""
     free!(obj::AbstractEnvironment)

Free the local tensors in `El[Center[1] + : end]` and `Er[1 : Center[2] - 1]`.
"""
function free!(obj::SimpleEnvironment)
     obj.El[obj.Center[1]+1:end] .= nothing
     obj.Er[1:obj.Center[2]-1] .= nothing
     return obj
end
function free!(obj::SparseEnvironment{L,N,T,StoreMemory}) where {L,N,T}
     for i in filter(i -> isassigned(obj.El, i), obj.Center[1]+1:L)
          obj.El[i] .= nothing
     end
     for i in filter(i -> isassigned(obj.Er, i), 1:obj.Center[2]-1)
          obj.Er[i] .= nothing
     end
     return obj
end
function free!(obj::SparseEnvironment{L,N,T,StoreDisk}) where {L,N,T}
     # freeing memory is not needed if the local tensors are stored in disk
     return obj
end


