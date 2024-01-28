"""
     inner(A::DenseMPS, B::DenseMPS)

Return the inner product `⟨A, B⟩` between MPS/MPO `A` and `B`.
"""
function inner(A::DenseMPS{L}, B::DenseMPS{L}) where {L}
     @assert codomain(A[1])[1] == codomain(B[1])[1] && domain(A[end])[end] == domain(B[end])[end]

     Env = Environment(A', B)
     return scalar!(Env; normalize = false, tmp = true)
end