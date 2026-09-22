"""
     bonddim(V::ElementarySpace) -> (D, DD)
     bonddim(A::AbstractTensorMap, idx::Integer) -> (D, DD)

Return the dimension of a given space or index of tensor `A`.

`D` is the number of multiplets, `DD` is the number of equivalent no symmetry states. Note `D == DD` for abelian groups.
"""
bonddim(V::ElementarySpace) = (reduceddim(V), dim(V))
bonddim(A::AbstractTensorMap, idx::Integer) = bonddim(space(A, idx))
