"""
     mutable struct MPS{L, T <:Union{Float64, ComplexF64}, C} <: DenseMPS{L, T}
          const A::AbstractVector{AbstractMPSTensor}
          const Center::Vector{Int64} 
          c::T 
     end

Concrete type of MPS, where `L` is the length, `T == Float64` or `ComplexF64` is the number type of local tensors.

`C <: AbstractStoreType` is the type to determine the storage of local tensors, usually `C == StoreMemory`. We can use package `SerializedElementArrays` to store local tensors in disk, in which case `C == StoreDisk`.

# Fields
     const A::AbstractVector{AbstractMPSTensor} 
Length L vector to store the local tensors. Note the vector `A` is immutable while the local tensors in it are mutable.
     
     const Center::Vector{Int64} 
Length 2 vector to label the canonical form. `[a, b]` means left-canonical from `1` to `a-1` and right-canonical from `b+1` to `L`.

     c::T
The global coefficient, i.e. we represented a MPS with L local tensors and an additional scalar `c`, in order to avoid too large/small local tensors. 

# Constructors
     MPS{L, T}(A::AbstractVector{<:AbstractMPSTensor}, Center::Vector{Int64} = [1, L], c::T = one(T))
Standard constructor. Note we will promote all local tensors if `T == ComplexF64` while the input local tensors in `A` are of `Float64`.

     MPS(A::AbstractVector{<:AbstractMPSTensor}, args...)
Deduce type parameters via `L = length(A)` and `T = typeof(c)`.  
For default `c` cases, assume `T = Float64` if all local tensors are real, otherwise `T = ComplexF64`.

     MPS(A::AbstractVector{<:AbstractTensorMap}, args...)
Automatically convert local tensors to warpper type `MPSTensor`.

     MPS{L, T}() 
     MPS(L, T = Float64)
Initialize an MPS{L, T} with the local tensors `A` to be filled. Note we initialize `Center = [1, L]` and `c = one(T)`.

Newest update, kwargs `disk::Bool = false` can be added to each constructor to control how to store the local tensors.
"""
mutable struct MPS{L, T <:Union{Float64, ComplexF64}, C} <: DenseMPS{L, T}
     const A::AbstractVector{AbstractMPSTensor}
     const Center::Vector{Int64} 
     c::T 

     function MPS{L, T}(; disk::Bool = false) where {L, T}

          A = Vector{AbstractMPSTensor}(undef, L)
          if disk 
               A = SerializedElementArrays.disk(A)
               C = StoreDisk
          else
               C = StoreMemory
          end
          return new{L, T, C}(A, [1, L], one(T))
     end
     MPS(L::Int64, T::Type{<:Union{Float64, ComplexF64}} = Float64; kwargs...) = MPS{L, T}(;kwargs...)

     function MPS{L, T}(A::AbstractVector{<:AbstractMPSTensor},
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
     MPS{L, T}(A::AbstractVector{<:AbstractMPSTensor}, c::T = one(T); kwargs...) where {L, T} = MPS{L, T}(A, [1, L], c; kwargs...)

     function MPS(A::AbstractVector{<:AbstractMPSTensor}, Center::Vector{Int64}, c::T; kwargs...) where T
          L = length(A)
          return MPS{L, T}(A, Center, c; kwargs...)
     end
     function MPS(A::AbstractVector{<:AbstractMPSTensor}, c::T; kwargs...) where T
          L = length(A)
          return MPS{L, T}(A, [1, L], c; kwargs...)
     end
     function MPS(A::AbstractVector{<:AbstractMPSTensor}, Center::Vector{Int64} = [1, length(A)]; kwargs...)
          L = length(A)
          T = mapreduce(eltype, promote_type, A)
          return MPS{L, T}(A, Center, one(T); kwargs...)          
     end

     function MPS(A::AbstractVector{<:AbstractTensorMap}, args...; kwargs...) 
          return MPS(convert(Vector{MPSTensor}, A), args...; kwargs...)
     end
     function MPS{L, T}(A::AbstractVector{<:AbstractTensorMap}, args...; kwargs...) where {L, T}
          return MPS{L, T}(convert(Vector{AbstractMPSTensor}, A), args...; kwargs...)
     end

end

"""
     randMPS([::Type{T},]  
          pspace::Vector{VectorSpace},
          aspace::Vector{VectorSpace};
          kwargs...) -> MPS{L}

Generate a length `L` random MPS with given length `L` vector `pspace` and `aspace`. `T = Float64`(default) or `ComplexF64` is the number type. Note the canonical center is initialized to the first site.

     randMPS([::Type{T},] L::Int64, pspace::VectorSpace, apsace::VectorSpace; kwargs...) -> MPS{L}

Assume the same `pspace` and `aspace`, except for the boundary bond, which is assumed to be trivial.  
"""
function randMPS(::Type{T}, pspace::AbstractVector{<:VectorSpace}, aspace::AbstractVector{<:VectorSpace}; kwargs...) where T <: Union{Float64, ComplexF64}
     
     @assert (L = length(pspace)) == length(aspace)

     obj = MPS(L, T; kwargs...)
     for si = 1:L
          if si == L
               obj[si] = randisometry(T, aspace[si]⊗pspace[si], trivial(pspace[si]); kwargs...)
          else
               obj[si] = randisometry(T, aspace[si]⊗pspace[si], aspace[si+1]; kwargs...)
          end
     end
     canonicalize!(obj, L)
     canonicalize!(obj, 1)
     return normalize!(obj)
end
function randMPS(::Type{T}, L::Int64, pspace::VectorSpace, aspace::VectorSpace; kwargs...) where T <: Union{Float64, ComplexF64}
     return randMPS(T, repeat([pspace,], L), vcat([trivial(pspace),], repeat([aspace,], L-1)); kwargs...)
end
function randMPS(::Type{T}, pspace::VectorSpace, aspace::AbstractVector{<:VectorSpace}; kwargs...) where T <: Union{Float64, ComplexF64}
     L = length(aspace)
     return randMPS(T, repeat([pspace,], L), aspace; kwargs...)
end
function randMPS(A::Any, args...; kwargs...)
     @assert !isa(A, DataType)
     return randMPS(Float64, A, args...; kwargs...)
end

similar(T_new::Type{<:Union{Float64, ComplexF64}}, A::MPS{L, T, StoreMemory}) where {L, T} = MPS{L, T_new}()
similar(T_new::Type{<:Union{Float64, ComplexF64}}, A::MPS{L, T, StoreDisk}) where {L, T} = MPS{L, T_new}(;disk = true)
similar(A::MPS{L, T}) where {L, T} = similar(T, A)
#TODO deepcopy for StoreDisk MPS














