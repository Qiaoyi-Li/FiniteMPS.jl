function mul!(C::T, A::SparseMPO, B::T, α::Number, β::Number; kwargs...) where {L,T<:DenseMPS{L}}
     # Note C should not be changed during iterating, so we actually store the result in D, and finially overwrite C

     D = get(kwargs, :init, α < β ? deepcopy(C) : deepcopy(B))


end