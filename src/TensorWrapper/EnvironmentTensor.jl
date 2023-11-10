"""
     abstract type AbstractEnvironmentTensor <: AbstractTensorWrapper

Wrapper type for left and right environment tensors.
"""
abstract type AbstractEnvironmentTensor <: AbstractTensorWrapper end


"""
     struct LocalLeftTensor{R} <: AbstractEnvironmentTensor
          A::AbstractTensorMap
          tag::NTuple{R, String}
     end

Wrapper type for rank-R left environment tenosr, with an additional field `tag` to distinguish legs of different channels.

Convention (' marks codomain):
     
      --R
     |
     | --R-1(') 
     El  ⋮
     | --2(')
     |
      --1'

# Constructors
     LocalLeftTensor(A::AbstractTensorMap [, tag::NTuple{R, String}])
Default tag = ("", "", ..., "").

     LocalLeftTensor{R}(A::AbstractTensorMap)
Used for automatic converting, only support default tag.
"""
struct LocalLeftTensor{R} <: AbstractEnvironmentTensor
     A::AbstractTensorMap
     tag::NTuple{R, String}
     function LocalLeftTensor(A::AbstractTensorMap, tag::NTuple{R, String}) where R
          @assert rank(A) == R
          return new{R}(A, tag)
     end
     function LocalLeftTensor{R}(A::AbstractTensorMap) where R
          # empty tag
          @assert rank(A) == R
          tag = Tuple(repeat(["",], R))
          return new{R}(A, tag)
     end
     function LocalLeftTensor(A::AbstractTensorMap)
          R = rank(A)
          return LocalLeftTensor{R}(A)
     end
end

"""
     struct LocalRightTensor{R} <: AbstractEnvironmentTensor
          A::AbstractTensorMap
          tag::NTuple{R, String}
     end

Wrapper type for rank-R right environment tenosr, with an additional field `tag` to distinguish legs of different channels.

Convention (' marks codomain):
     
         1'--
             |
       2(')--| 
       ⋮     Er  
     R-1(')--| 
             |
          R--

# Constructors
     LocalRigthTensor(A::AbstractTensorMap [, tag::NTuple{R, String}])
Default tag = ("", "", ..., "").

     LocalRightTensor{R}(A::AbstractTensorMap)
Used for automatic converting, only support default tag.
"""
struct LocalRightTensor{R} <: AbstractEnvironmentTensor
     A::AbstractTensorMap
     tag::NTuple{R, String}
     function LocalRightTensor(A::AbstractTensorMap, tag::NTuple{R, String}) where R
          @assert rank(A) == R
          return new{R}(A, tag)
     end
     function LocalRightTensor{R}(A::AbstractTensorMap) where R
          # empty tag
          @assert rank(A) == R
          tag = Tuple(repeat(["",], R))
          return new{R}(A, tag)
     end
     function LocalRightTensor(A::AbstractTensorMap)
          R = rank(A)
          return LocalRightTensor{R}(A)
     end
end

"""
     const SimpleLeftTensor = Union{Nothing, LocalLeftTensor} 

Type of left environment tensor of simple MPO, i.e. a channel of sparse MPO.
"""
const SimpleLeftTensor = Union{Nothing, LocalLeftTensor} 

"""
     const SimpleRightTensor = Union{Nothing, LocalRightTensor} 

Type of right environment tensor of simple MPO, i.e. a channel of sparse MPO.
"""
const SimpleRightTensor = Union{Nothing, LocalRightTensor} 

"""
     const SparseLeftTensor = Vector{SimpleLeftTensor}

Type of left environment tensor of sparse MPO.
"""
const SparseLeftTensor = Vector{SimpleLeftTensor}

"""
     const SparseRightTensor = Vector{SimpleRightTensor}

Type of right environment tensor of sparse MPO.
"""
const SparseRightTensor = Vector{SimpleRightTensor}


"""
     *(A::LocalLeftTensor{R}, B::LocalRightTensor{R}) -> ::Number

Contract a left environment tenosr and a right environment tensor with the same virtual spaces to get a scalar.
"""
function *(A::LocalLeftTensor{R}, B::LocalRightTensor{R}) where R
     if R ≥ 4
          # up to a permutation in some cases
          @assert A.tag[2:R-1] == B.tag[2:R-1]
     end
     return scalar(TensorOperations.tensorcontract(A.A, 1:R, B.A, reverse(1:R)))
end

""" 
     *(El::LocalLeftTensor, A::MPSTensor)

Conctract a MPS tensor to a left environment tensor.
"""
function *(El::LocalLeftTensor{R₁}, A::MPSTensor{R₂}) where {R₁, R₂}
     return permute(El.A, Tuple(1:R₁-1), (R₁,)) * permute(A.A, (1,), Tuple(2:R₂))
end


