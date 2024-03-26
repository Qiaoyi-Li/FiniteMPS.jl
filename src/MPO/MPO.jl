"""
     mutable struct MPO{L, T <:Union{Float64, ComplexF64}, C} <: DenseMPS{L, T}
          const A::AbstractVector{AbstractMPSTensor}
          const Center::Vector{Int64} 
          c::T 
     end

All the fields and constructors are exactly the same to those of `MPS`, we redefine the type `MPO` just for using multiple dispacth when implementing the algebra between `MPS` and `MPO`. 

Details of constructors please see `MPS`. 
"""
mutable struct MPO{L, T <:Union{Float64, ComplexF64}, C} <: DenseMPS{L, T}
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

similar(T_new::Type{<:Union{Float64, ComplexF64}}, A::MPO{L, T, StoreMemory}) where {L, T} = MPO{L, T_new}()
similar(T_new::Type{<:Union{Float64, ComplexF64}}, A::MPO{L, T, StoreDisk}) where {L, T} = MPO{L, T_new}(;disk = true)
similar(A::MPO{L, T}) where {L, T} = similar(T, A)

"""
     identityMPO(::Type{T} = Float64, L::Int64, pspace::AbstractVector; kwargs...)

Construct an identity MPO where the physical spaces are informed by a length `L` vertor of `VectorSpace`.

     identityMPO(::Type{T} = Float64, L::Int64, pspace::VectorSpace; kwargs...)
Assume the all the physical spaces are the same.

     identityMPO(obj::DenseMPS{L, T}; kwargs...)
Deduce the scalar type `T` and physical spaces from a MPS/MPO.

# Kwargs
     disk::Bool = false
     bspace::VectorSpace
Space of left boundary bond, default = trivial space.
"""
function identityMPO(::Type{T}, L::Int64, pspace::AbstractVector; kwargs...) where T <: Union{Float64, ComplexF64}
     @assert length(pspace) == L

     obj = MPO(L, T; disk = get(kwargs, :disk, false))  
     aspace = trivial(pspace[1])
     bspace = get(kwargs, :bspace, aspace)
     obj[1] = permute(isometry(bspace, aspace) ⊗ isometry(pspace[1], pspace[1]), ((1, 2), (4, 3)))
     for si in 2:L
          obj[si] = permute(isometry(aspace, aspace) ⊗ isometry(pspace[si], pspace[si]), ((1, 2), (4, 3)))  
     end

     canonicalize!(obj, L)
     canonicalize!(obj, 1)
     return obj
end
function identityMPO(::Type{T}, L::Int64, pspace::VectorSpace; kwargs...) where T <: Union{Float64, ComplexF64}
     return identityMPO(T, L, [pspace for i in 1:L]; kwargs...)
end
identityMPO(L::Int64, pspace::Union{AbstractVector, VectorSpace}; kwargs...) = identityMPO(Float64, L, pspace; kwargs...) # default = Float64

function identityMPO(obj::DenseMPS{L, T}; kwargs...) where {L, T}
     return identityMPO(T, L, [codomain(obj[si])[2] for si in 1:L]; kwargs...)
end