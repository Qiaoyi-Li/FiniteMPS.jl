"""
     struct LeftPreFuseTensor{R} <: AbstractEnvironmentTensor
          A::AbstractTensorMap
     end

Wrapper type for the prefused left environment tensors, i.e., the combination of a left environment tensor and a local operator.
     
Convention (' marks codomain): 

      --R-1  R
     |       |
     El------Hl--(R-2)' 
     |       |
      --1'   2' 
"""
struct LeftPreFuseTensor{R} <: AbstractEnvironmentTensor
     A::AbstractTensorMap
     function LeftPreFuseTensor{R}(A::AbstractTensorMap) where {R}
          @assert R == rank(A) ≥ 4
          if rank(A, 2) != 2
               A = permute(A, Tuple(1:R-2), (R - 1, R))
          end
          return new{R}(A)
     end
     function LeftPreFuseTensor(A::AbstractTensorMap)
          R = rank(A)
          return LeftPreFuseTensor{R}(A)
     end
end

# TODO right version

const SparseLeftPreFuseTensor = Vector{LeftPreFuseTensor}