"""
     abstract type AbstractEnvironment{L}

Abstract type of all multi-layer environments. 
"""
abstract type AbstractEnvironment{L} end

length(::AbstractEnvironment{L}) where {L} = L
# note we prohibit to change the MPS obj of an environment, hence we will not provide setindex
for func in (:getindex, :lastindex, :iterate, :keys)
     @eval Base.$func(obj::AbstractEnvironment, args...) = $func(obj.layer, args...)
end

"""
     Center(obj::AbstractEnvironment) -> Vector (length 2)

Interface of environment, return the info of canonical center. `Center = [a, b]` means El[1:a] and Er[b:end] are valid.
"""
Center(obj::AbstractEnvironment) = obj.Center

"""
     struct SimpleEnvironment{L, N, T<:NTuple{N, AbstractMPS{L}}, C<:AbstractStoreType} <: AbstractEnvironment{L}
          layer::T
          El::AbstractVector{SimpleLeftTensor}
          Er::AbstractVector{SimpleRightTensor}
          Center::Vector{Int64} 
     end

Type of all environments whose local left/right tensors are simple, i.e. a single tensor instead of a vector storing several tensors.

`Center = [a, b]` means El[1:a] and Er[b:end] are valid.
"""
struct SimpleEnvironment{L,N,T<:NTuple{N,AbstractMPS{L}},C<:AbstractStoreType} <: AbstractEnvironment{L}
     layer::T
     El::AbstractVector{SimpleLeftTensor}
     Er::AbstractVector{SimpleRightTensor}
     Center::Vector{Int64}
     function SimpleEnvironment(layer::T,
          El::AbstractVector{SimpleLeftTensor},
          Er::AbstractVector{SimpleRightTensor},
          Center::Vector{Int64}; disk::Bool=false) where {L,N,T<:NTuple{N,AbstractMPS{L}}}
          if disk
               C = StoreDisk
               El =  SerializedElementArrays.disk(El)
               Er =  SerializedElementArrays.disk(Er)
          else
               C = StoreMemory
               El = convert(Vector{SimpleLeftTensor}, El)
               Er = convert(Vector{SimpleRightTensor}, Er)
          end
          return new{L,N,T,C}(layer, El, Er, Center)
     end
end

"""
     struct SparseEnvironment{L, N, T<:NTuple{N, AbstractMPS{L}}, C<:AbstractStoreType} <: AbstractEnvironment{L}
          layer::T
          El::AbstractVector{SparseLeftTensor}
          Er::AbstractVector{SparseRightTensor}
          Center::Vector{Int64} 
     end

Type of all environments whose local left/right tensors are vectors storing several tensors, usually due to a `SparseMPO` in a layer.

`Center = [a, b]` means El[1:a] and Er[b:end] are valid.
"""
struct SparseEnvironment{L,N,T<:NTuple{N,AbstractMPS{L}},C<:AbstractStoreType} <: AbstractEnvironment{L}
     layer::T
     El::AbstractVector{SparseLeftTensor}
     Er::AbstractVector{SparseRightTensor}
     Center::Vector{Int64}
     function SparseEnvironment(layer::T,
          El::AbstractVector{SparseLeftTensor},
          Er::AbstractVector{SparseRightTensor},
          Center::Vector{Int64}; disk::Bool=false) where {L,N,T<:NTuple{N,AbstractMPS{L}}}
          if disk
               C = StoreDisk
               El =  SerializedElementArrays.disk(El)
               Er =  SerializedElementArrays.disk(Er)
          else
               C = StoreMemory
               El = convert(Vector{SparseLeftTensor}, El)
               Er = convert(Vector{SparseRightTensor}, Er)
          end
          return new{L,N,T,C}(layer, El, Er, Center)
     end
end

"""
     Environment(M::AbstractMPS{L}...; kwargs...)

Generic constructor of environments. 

Generate an environment object according to the input MPS/MPO objects. The boundary environment tensor, i.e. `El[1]` and `Er[L]` will be initialized. 

# Kwargs
     disk::Bool = false
Store the local environment tensor in disk(`true`) or in memory(`false`).

     El::AbstractTensorMap
     Er::AbstractTensorMap
Initialize boundary `El` or `Er` manually. Default value is generated by function `_defaultEl` or `_defaultEr`, respectively.
"""
function Environment(M::AbstractMPS{L}...; kwargs...) where L
     layer = M
     Center = [0, L + 1]
     if any(issparse, layer)
          El = Vector{SparseLeftTensor}(undef, L)
          Er = Vector{SparseRightTensor}(undef, L)
          obj = SparseEnvironment(layer, El, Er, Center; kwargs...)
     else
          El = Vector{SimpleLeftTensor}(undef, L)
          Er = Vector{SimpleRightTensor}(undef, L)
          obj = SimpleEnvironment(layer, El, Er, Center; kwargs...)
     end

     _initializeEnv!(obj)
     return obj
end

function show(io::IO, obj::AbstractEnvironment)
     # only contain the memory of El and Er
     memory = Base.format_bytes(Base.summarysize(obj.El) + Base.summarysize(obj.Er))
     isa(obj, SparseEnvironment) ? print(io, "Sparse ") : print(io, "Simple ")
     println(io, "Environment: Center = $(obj.Center), total memory = $memory")
     # print MPS objs from top to bottom
     for i in reverse(eachindex(obj.layer))
          print(io, "[$i] ")
          show(io, obj.layer[i])
     end
end



