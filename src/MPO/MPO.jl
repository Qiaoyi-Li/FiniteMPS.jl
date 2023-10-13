"""
     mutable struct MPO{L, T <:Union{Float64, ComplexF64}, C} <: DenseMPS{L}
          const A::AbstractVector{AbstractMPSTensor}
          const Center::Vector{Int64} 
          c::T 
     end

All the fields and constructors are exactly the same to those of `MPS`, we redefine the type `MPO` just for using multiple dispacth when implementing the algebra between `MPS` and `MPO`. 

Details of constructors please see `MPS`. 
"""
mutable struct MPO{L, T <:Union{Float64, ComplexF64}, C} <: DenseMPS{L}
     const A::AbstractVector{AbstractMPSTensor}
     const Center::Vector{Int64} 
     c::T 

     function MPO{L, T}(; disk::Bool = false) where {L, T}

          A = Vector{AbstractMPSTensor}(undef, L)
          if disk 
               A = SerializedElementArrays.disk(A)
               C = StoreDisk
          else
               C = StoreMemory
          end
          return new{L, T, C}(A, [1, L], one(T))
     end
     MPO(L::Int64, T::Type{<:Union{Float64, ComplexF64}} = Float64; kwargs...) = MPO{L, T}(;kwargs...)

     function MPO{L, T}(A::AbstractVector{<:AbstractMPSTensor},
          Center::Vector{Int64},
          c::T = one(T);
          disk::Bool = false) where {L, T}
          @assert length(A) == L     
          @assert length(Center) == 2 && Center[1] ≥ 1 && Center[2] ≤ L
          @assert T ∈ [Float64, ComplexF64]
          if T == ComplexF64  # promote each A
               for i = 1:L
                    eltype(A[i]) != T && (A[i] *= one(T))
               end
          end
          
          if eltype(A) != AbstractMPSTensor
               A = convert(Vector{AbstractMPSTensor}, A)
          end

          if disk 
               A = SerializedElementArrays.disk(A)
               C = StoreDisk
          else
               C = StoreMemory
               if A isa SerializedElementArrays.SerializedElementVector
                    A = convert(Vector{AbstractMPSTensor}, A)
               end
          end
          return new{L, T, C}(A, Center, c)
     end
     MPO{L, T}(A::AbstractVector{<:AbstractMPSTensor}, c::T = one(T); kwargs...) where {L, T} = MPS{L, T}(A, [1, L], c; kwargs...)

     function MPO(A::AbstractVector{<:AbstractMPSTensor}, Center::Vector{Int64}, c::T; kwargs...) where T
          L = length(A)
          return MPO{L, T}(A, Center, c; kwargs...)
     end
     function MPO(A::AbstractVector{<:AbstractMPSTensor}, c::T; kwargs...) where T
          L = length(A)
          return MPO{L, T}(A, [1, L], c; kwargs...)
     end
     function MPO(A::AbstractVector{<:AbstractMPSTensor}, Center::Vector{Int64} = [1, length(A)]; kwargs...)
          L = length(A)
          T = mapreduce(eltype, promote_type, A)
          return MPO{L, T}(A, Center, one(T); kwargs...)          
     end

     function MPO(A::AbstractVector{<:AbstractTensorMap}, args...; kwargs...) 
          return MPO(convert(Vector{MPSTensor}, A), args...; kwargs...)
     end
     function MPO{L, T}(A::AbstractVector{<:AbstractTensorMap}, args...; kwargs...) where {L, T}
          return MPO{L, T}(convert(Vector{AbstractMPSTensor}, A), args...; kwargs...)
     end

end


# # initialize an identity MPO
# # TODO from pspace, from MPS/MPO ...
# function identityMPO(::Type{T}, L::Int64, pspace::VectorSpace) where T <: Union{Float64, ComplexF64}
#      A = Vector{MPSTensor}(undef, L) 
#      aspace = trivial(pspace)
#      for si in eachindex(A)
#           A[si] = TensorMap(ones, T, aspace⊗pspace, pspace⊗aspace)
#      end
#      obj = MPO(A)
#      # canonicalize and normalize
#      canonicalize!(obj, L)
#      canonicalize!(obj, 1)
#      return obj
# end
# identityMPO(L::Int64, pspace::VectorSpace) = identityMPO(Float64, L, pspace) # default = Float64

