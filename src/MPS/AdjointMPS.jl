"""
     struct AdjointMPS{L} <: AbstractMPS{L}
          parent::DenseMPS{L}
     end

Lazy wrapper type for adjoint of MPS.

     adjoint(::DenseMPS) -> ::AdjointMPS
     adjoint(::AdjointMPS) -> ::DenseMPS

Functions to be directly propagated to the parent:

     lastindex, length, keys, norm, normalize!, Center, iterate, canonicalize!

Functions to be propagated to the parent with some adaptations:

     getindex, setindex!, coef
"""
struct AdjointMPS{L} <: AbstractMPS{L}
     parent::DenseMPS{L}
end
adjoint(A::DenseMPS) = AdjointMPS(A)
adjoint(A::AdjointMPS) = A.parent

function show(io::IO, obj::AdjointMPS)
     print(io, "Adjoint of ")
     show(io, obj.parent)
end

# apply lazy adjoint when obtaining the local tensors
getindex(obj::AdjointMPS, inds...) = getindex(obj.parent, inds...)'
setindex!(obj::AdjointMPS, X, inds...) = setindex!(obj.parent, X', inds...)
"""
     coef(obj::AdjointMPS) = coef(obj.parent)'
"""
coef(obj::AdjointMPS) = coef(obj.parent)'

# some functions to be directly propagated to the parent
for func in (:lastindex, :length, :keys,:norm, :normalize!, :Center, :iterate, :canonicalize!, :scalartype)
     @eval $func(obj::AdjointMPS, args...) = $func(obj.parent, args...)
end

