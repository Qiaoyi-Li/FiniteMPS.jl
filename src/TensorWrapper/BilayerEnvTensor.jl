"""
     struct BilayerLeftTensor{R₁, R₂} <: AbstractEnvironmentTensor
          A::AbstractTensorMap
     end

Wrapper type of left environment tensor especially for calculating ITP. Use domain and codomain to distinguish the horizontal bonds of two layers.

Convention (' marks codomain):
     
      --R₂
     |   ⋮
     | --R₁+1 
     El  
     | --R₁'
     |   ⋮
      --1'

# Constructors
     BilayerLeftTensor(A::AbstractTensorMap)
     BilayerLeftTensor{R₁, R₂}(A::AbstractTensorMap)
"""
struct BilayerLeftTensor{R₁, R₂} <: AbstractEnvironmentTensor
     A::AbstractTensorMap
     function BilayerLeftTensor{R₁, R₂}(A::AbstractTensorMap) where {R₁, R₂}
          @assert rank(A, 1) == R₁ && rank(A, 2) == R₂
          return new{R₁,R₂}(A)
     end
     function BilayerLeftTensor(A::AbstractTensorMap)
          R₁ = rank(A, 1)
          R₂ = rank(A, 2)
          return BilayerLeftTensor{R₁, R₂}(A)
     end
end

"""
     struct BilayerRightTensor{R} <: AbstractEnvironmentTensor
          A::AbstractTensorMap
     end

Wrapper type of right environment tensor especially for calculating ITP. Use domain and codomain to distinguish the horizontal bonds of two layers.

Convention (' marks codomain):
     
         1'--
         ⋮   |
        R₁'--| 
             Er  
       R₁+1--| 
         ⋮   |
         R₂--

# Constructors
     BilayerRightTensor(A::AbstractTensorMap)
     BilayerRightTensor{R₁, R₂}(A::AbstractTensorMap)
"""
struct BilayerRightTensor{R₁, R₂} <: AbstractEnvironmentTensor
     A::AbstractTensorMap
     function BilayerRightTensor{R₁,R₂}(A::AbstractTensorMap) where {R₁, R₂}
          @assert rank(A, 1) == R₁ && rank(A, 2) == R₂
          return new{R₁,R₂}(A)
     end
     function BilayerRightTensor(A::AbstractTensorMap)
          R₁ = rank(A, 1)
          R₂ = rank(A, 2)
          return BilayerRightTensor{R₁, R₂}(A)
     end
end


"""
     *(A::BilayerLeftTensor{R₁, R₂}, B::BilayerRightTensor{R₂,R₁}) -> ::Number

Contract a left environment tenosr and a right environment tensor with the same virtual spaces to get a scalar.
"""
function *(A::BilayerLeftTensor{R₁,R₂}, B::BilayerRightTensor{R₂,R₁}) where {R₁, R₂}
     R = R₁ + R₂
     return scalar(TensorOperations.tensorcontract(A.A, 1:R, B.A, reverse(1:R)))
end
