"""
     struct AdjointMPSTensor{R} <: AbstractMPSTensor
          A::AbstractTensorMap
     end

Lazy wrapper type for tensors of adjoint MPS.  

     adjoint(::MPSTensor) -> ::AdjointMPSTensor
     adjoint(::AdjointMPSTensor) -> ::MPSTensor

Convention (' marks codomain): 

                       3              4                   R
                       |              |                   |
     2-- A --1'     2--A--1'       3--A--2'       (R-1)-- A  --(R-2)'        
                                      |                 / | \\
                                      1'              1' ... (R-3)'
"""
struct AdjointMPSTensor{R} <: AbstractMPSTensor
     A::AbstractTensorMap
end
adjoint(A::MPSTensor{R}) where R = AdjointMPSTensor{R}(A.A')
adjoint(A::AdjointMPSTensor)::MPSTensor = A.A'

