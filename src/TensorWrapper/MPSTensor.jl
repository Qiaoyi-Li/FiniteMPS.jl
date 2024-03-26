"""
    abstract type AbstractMPSTensor <: AbstractTensorWrapper

Elements of MPS, note a MPO is nothing but a MPS with rank-4 tensors, hence we using this type to deal with both MPS and MPO    
"""
abstract type AbstractMPSTensor <: AbstractTensorWrapper end
Base.convert(::Type{AbstractMPSTensor}, A::AbstractTensorMap) = MPSTensor(A)

"""
     struct MPSTensor{R} <: AbstractMPSTensor
          A::AbstractTensorMap
     end 
          
Wrapper type for rank-R MPS local tensors.

Convention (' marks codomain): 

          3 ... (R-1)
          \\ | /  
     1'--   A  ---R         1'-- A -- 2
            | 
            2'   
In particular, R == 2 for bond tensor.

# Constructors
     MPSTensor(::AbstractTensorMap) 
     MPSTensor{R}(::AbstractTensorMap)
"""
struct MPSTensor{R} <: AbstractMPSTensor
     A::AbstractTensorMap
     function MPSTensor{R}(A::AbstractTensorMap) where {R}
          @assert R == rank(A) ≥ 2
          if R ≥ 3 && rank(A, 1) != 2
               A = permute(A, ((1, 2), Tuple(3:R)))
          end
          return new{R}(A)
     end
     function MPSTensor(A::AbstractTensorMap)
          R = rank(A)
          return MPSTensor{R}(A)
     end
end

"""
     *(A::MPSTensor, B::MPSTensor) -> ::AbstractTensorMap

Contract the virtual bond between 2 neighbor local tensors.
"""
function *(A::MPSTensor{R₁}, B::MPSTensor{R₂}) where {R₁,R₂}
     C = TensorOperations.tensorcontract(
          (Tuple(1:R₁-1), Tuple(R₁:(R₁+R₂-2))),
          A.A, (Tuple(1:R₁-1), (R₁,)), :N,
          B.A, ((1,), Tuple(2:R₂)), :N)
     return C
end
promote_rule(::Type{<:MPSTensor}, ::Type{<:AbstractTensorMap}) = MPSTensor
*(A::Union{MPSTensor,AbstractTensorMap}, B::Union{MPSTensor,AbstractTensorMap}) = *(promote(A, B)...)

"""
     leftorth(A::MPSTensor; 
          trunc = notrunc(),
          kwargs...) -> Q::AbstractTensorMap, R::AbstractTensorMap, info::BondInfo

Left canonicalize a on-site MPS tensor. 

If `trunc = notrunc()`, use `TensorKit.leftorth`, otherwise, use `TensorKit.tsvd`. Propagate `kwargs` to the TensorKit functions.
"""
function leftorth(A::MPSTensor{R₁}; trunc=notrunc(), kwargs...) where {R₁}
     if trunc == notrunc()
          Q, R = leftorth(A.A, Tuple(1:R₁-1), (R₁,); kwargs...)
          return Q, R, BondInfo(Q, :R)
     else
          u, s, vd, info =  tsvd(A, Tuple(1:R₁-1), (R₁,); trunc=trunc, kwargs...)
          return u, s * vd, info
     end

end

"""
     rightorth(A::MPSTensor;
          trunc = notrunc(),
          kwargs...) -> L::AbstractTensorMap, Q::AbstractTensorMap, info::BondInfo

Right canonicalize a on-site MPS tensor. 

If `trunc = notrunc()`, use `TensorKit.rightorth`, otherwise, use `TensorKit.tsvd`. Propagate `kwargs` to the TensorKit functions.
"""
function rightorth(A::MPSTensor{R₂}; trunc=notrunc(), kwargs...) where {R₂}
     if trunc == notrunc()
          L, Q = rightorth(A.A, (1,), Tuple(2:R₂); kwargs...)
          return L, Q, BondInfo(Q, :L)
     else
          u, s, vd, info = tsvd(A, (1,), Tuple(2:R₂); trunc=trunc, kwargs...)
          return u * s, vd, info
     end
end












