"""
     dim(A::AbstractTensorMap, idx::Int64) -> (D, DD)::NTuple{2, Int64}

Return the dimension of a given index of tensor `A`.

`D` is the number of multiplets, `DD` is the number of equivalent no symmetry states. Note `D == DD` for abelian groups.
"""
function dim(A::AbstractTensorMap{<:Union{Float64, ComplexF64}, F}, idx::Int64) where F<:GradedSpace
     rcod = numout(A)
     if idx ≤ rcod[1]
          D = mapreduce(i -> codomain(A).spaces[idx].dims[i], +, eachindex(codomain(A).spaces[idx].dims))
          DD = dim(codomain(A).spaces[idx])
     else
          D = mapreduce(i -> domain(A).spaces[idx - rcod].dims[i], +, eachindex(domain(A).spaces[idx - rcod].dims))
          DD = dim(domain(A).spaces[idx - rcod])
     end
    return D, DD
end

function dim(A::AbstractTensorMap{<:Union{Float64, ComplexF64} ,F}, idx::Int64) where F<:Union{CartesianSpace,ComplexSpace}
     return dim(space(A, idx)), dim(space(A, idx))
end

numin(::Nothing) = 0
numout(::Nothing) = 0
numind(::Nothing) = 0

# implement usage A.dom[end] ...
Base.lastindex(V::ProductSpace) = typeof(V).parameters[2]

"""
     data(A::AbstractTensorMap) -> collection of data
     data(A::AdjointTensorMap) = data(A.parent)

Interface of `AbstractTensorMap`, return the data of a given tensor.
"""
data(A::AbstractTensorMap{<:Union{Float64, ComplexF64}, <:GradedSpace}) = A.data
data(A::AbstractTensorMap{<:Union{Float64, ComplexF64}, <:Union{CartesianSpace,ComplexSpace}}) = [A.data,]
data(A::TensorKit.AdjointTensorMap) = data(A.parent)

# support nothing initialization when using @reduce
+(a::AbstractTensorMap, ::Nothing) = a
+(::Nothing, b::AbstractTensorMap) = b
axpy!(α::Number, ::Nothing, ::Nothing) = nothing
axpy!(α::Number, ::Nothing, A::AbstractTensorMap) = A
axpy!(α::Number, A::AbstractTensorMap, B::Nothing) = α * A
rmul!(::Nothing, ::Number) = nothing

dim(::Nothing, ::Int64) = (0, 0)
