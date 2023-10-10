
mutable struct MPO{L, T <:Union{Float64, ComplexF64}} <: AbstractMPS{L}
     const A::Vector{MPSTensor}  # MPO is nothing but MPS with rank-4 local tensors !
     const Center::Vector{Int64} # [a, b] means left-canonical from 1 to a-1 and right-canonical from L to b+1
     c::T # global coef
     # stardard constructor
     function MPO(A::Vector{M}) where M <: Union{MPSTensor, AbstractTensorMap}  
          T = mapreduce(eltype, promote_type, A)          
          if T == ComplexF64 
               # promote each A
               A .*= one(T)
          else
               @assert T == Float64 
          end
          L = length(A)

          return new{L, T}(A, [1, L], one(T))       
     end
end



# initialize an identity MPO
# TODO from pspace, from MPS/MPO ...
function identityMPO(::Type{T}, L::Int64, pspace::VectorSpace) where T <: Union{Float64, ComplexF64}
     A = Vector{MPSTensor}(undef, L) 
     aspace = trivial(pspace)
     for si in eachindex(A)
          A[si] = TensorMap(ones, T, aspace⊗pspace, pspace⊗aspace)
     end
     obj = MPO(A)
     # canonicalize and normalize
     canonicalize!(obj, L)
     canonicalize!(obj, 1)
     return obj
end
identityMPO(L::Int64, pspace::VectorSpace) = identityMPO(Float64, L, pspace) # default = Float64

