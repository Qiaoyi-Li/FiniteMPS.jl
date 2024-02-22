"""
     dim(A::AbstractTensorMap, idx::Int64) -> (D, DD)::NTuple{2, Int64}

Return the dimension of a given index of tensor `A`.

`D` is the number of multiplets, `DD` is the number of equivalent no symmetry states. Note `D == DD` for abelian groups.
"""
function dim(A::AbstractTensorMap{F}, idx::Int64) where F<:GradedSpace
     rcod = rank(A, 1)
     if idx ≤ rcod[1]
          D = mapreduce(i -> codomain(A).spaces[idx].dims[i], +, eachindex(codomain(A).spaces[idx].dims))
          DD = dim(codomain(A).spaces[idx])
     else
          D = mapreduce(i -> domain(A).spaces[idx - rcod].dims[i], +, eachindex(domain(A).spaces[idx - rcod].dims))
          DD = dim(domain(A).spaces[idx - rcod])
     end
    return D, DD
end

function dim(A::AbstractTensorMap{F}, idx::Int64) where F<:Union{CartesianSpace,ComplexSpace}
     return dim(space(A, idx)), dim(space(A, idx))
end

"""
     rank(A::AbstractTensorMap) -> ::Int64

Return the rank of a given tensor.

     rank(A::AbstractTensorMap, idx::Int64) -> ::Int64
Return the rank corresponding to codomain (`idx = 1`) or domain (`idx = 2`).
"""
function rank(A::AbstractTensorMap, idx::Int64)
     @assert idx ∈ [1, 2]
     idx == 1 && return typeof(codomain(A)).parameters[2]
     idx == 2 && return typeof(domain(A)).parameters[2]
end
rank(A::AbstractTensorMap) = rank(A, 1) + rank(A, 2)
rank(::Nothing, args...) = 0

# implement usage A.dom[end] ...
Base.lastindex(V::ProductSpace) = typeof(V).parameters[2]

"""
     data(A::AbstractTensorMap) -> collection of data
     data(A::AdjointTensorMap) = data(A.parent)

Interface of `AbstractTensorMap`, return the data of a given tensor.
"""
data(A::AbstractTensorMap{<:GradedSpace}) = A.data
data(A::AbstractTensorMap{<:Union{CartesianSpace,ComplexSpace}}) = [A.data,]
data(A::TensorKit.AdjointTensorMap) = data(A.parent)

# support nothing initialization when using @reduce
+(a::AbstractTensorMap, ::Nothing) = a
+(::Nothing, b::AbstractTensorMap) = b
axpy!(α::Number, ::Nothing, ::Nothing) = nothing
axpy!(α::Number, ::Nothing, A::AbstractTensorMap) = A
axpy!(α::Number, A::AbstractTensorMap, B::Nothing) = α * A
rmul!(::Nothing, ::Number) = nothing

dim(::Nothing, ::Int64) = (0, 0)
