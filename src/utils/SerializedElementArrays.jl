# add some methods for offical package SerializedElementArrays.jl

# make sure similar(::SerializedElementArrays) -> ::SerializedElementArrays
function Base.similar(d::SerializedElementArray{T, N}) where {T, N}
    return SerializedElementArrays.disk(Array{T, N}(undef, size(d)...))
end

# support usage like V[i:j] 
function Base.getindex(d::SerializedElementArray{T}, V::AbstractVector) where T
     return map(i -> convert(T, d[i]), V) 
 end