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

Wrapper type for rank-R left environment tensor, with an additional field `tag` to distinguish legs of different channels.

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

Wrapper type for rank-R right environment tensor, with an additional field `tag` to distinguish legs of different channels.

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

Contract a left environment tensor and a right environment tensor with the same virtual spaces to get a scalar.
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
     return permute(El.A, (Tuple(1:R₁-1), (R₁,))) * permute(A.A, ((1,), Tuple(2:R₂)))
end

""" 
     *(A::MPSTensor, Er::LocalRightTensor)

Conctract a MPS tensor to a right environment tensor.
"""
function *(A::MPSTensor{R₁}, Er::LocalRightTensor{R₂}) where {R₁, R₂}
     return permute(A.A, (Tuple(1:R₁-1), (R₁,))) * permute(Er.A, ((1,), Tuple(2:R₂)))
end

"""
     fuse(El::SimpleLeftTensor) -> iso::AbstractTensorMap
     fuse(El::SimpleRightTensor) -> iso::AbstractTensorMap
     
Return the isometry to fuse the top 2 legs.

     fuse(lsEl::SparseLeftTensor) -> Vector{AbstractTensorMap} 
     fuse(lsEl::SparseRightTensor) -> Vector{AbstractTensorMap}

Additionally embed the isometry to the direct sum space of all channels. 
"""
function fuse(lsEl::SparseLeftTensor)

     lsIso = map(fuse, lsEl)
     lsV = map(lsIso) do iso
          domain(iso)[1]
     end
     lsembed = oplusEmbed(lsV; reverse = true)
     return map(lsIso, lsembed) do iso, embed
          iso * embed
     end
end
function fuse(lsEr::SparseRightTensor)

     lsIso = map(fuse, lsEr)
     lsV = map(lsIso) do iso
          codomain(iso)[1]
     end
     lsembed = oplusEmbed(lsV)
     return map(lsIso, lsembed) do iso, embed
          embed * iso
     end
end
function fuse(El::LocalLeftTensor{2})
     return isometry(domain(El)[end], domain(El)[end])
end
function fuse(El::LocalLeftTensor{3})
     if rank(El, 1) == 1
          aspace = fuse(domain(El)[1], domain(El)[2])
          return isometry(domain(El)[1] ⊗ domain(El)[2], aspace)
     else
          aspace = fuse(codomain(El)[2]', domain(El)[1])
          return isometry(codomain(El)[2]' ⊗ domain(El)[1], aspace)
     end
end
function fuse(Er::LocalRightTensor{2})
     return isometry(codomain(Er)[1], codomain(Er)[1])
end
function fuse(Er::LocalRightTensor{3})
     if rank(Er, 1) == 1
          aspace = fuse(codomain(Er)[1], domain(Er)[1]')
          return isometry(aspace, codomain(Er)[1] ⊗ domain(Er)[1]')
     else
          aspace = fuse(codomain(Er)[1], codomain(Er)[2])
          return isometry(aspace, codomain(Er)[1] ⊗ codomain(Er)[2])
     end
end
fuse(::Nothing) = nothing
