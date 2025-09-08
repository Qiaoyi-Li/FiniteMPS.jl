"""
     module U₁SpinlessFermion

Prepare some commonly used objects for U₁ spinless fermions.

# Fields
     pspace::VectorSpace
Local `d = 2` Hilbert space.

     Z::TensorMap
Rank-`2` fermion parity operator `Z = (-1)^n`.

     n::TensorMap
Rank-`2` particle number operator.

     FdagF::NTuple{2, TensorMap}
Two rank-`3` operators of hopping `c^dag c`.

     FFdag::NTuple{2, TensorMap}
Two rank-`3` operators of hopping `c c^dag`.

     ΔdagΔ::NTuple{4, TensorMap}
Four operators of pairing `Δ^dag Δ`, where `Δᵢⱼ = cᵢcⱼ` is the pairing operator.  Rank = `(3, 4, 4, 3)`. Sign convention: `(i,j,k,l)` gives `cᵢ^dag cⱼ^dag cₖ cₗ = - cⱼ^dag cᵢ^dag cₖ cₗ = - Δᵢⱼ^dag Δₖₗ`.
"""
module U₁SpinlessFermion
using TensorKit
const pspace = Rep[U₁](-1 // 2 => 1, 1 // 2 => 1)

const Z = let
     Z = isometry(pspace, pspace)
     block(Z, Irrep[U₁](1 // 2)) .= -1
     Z
end

const n = let
     n = isometry(pspace, pspace)
     block(n, Irrep[U₁](-1 // 2)) .= 0
     n
end

# hopping term, FdagF
const FdagF = let
     aspace = Rep[U₁](1 => 1)
     Fdag = TensorMap(ones, pspace, pspace ⊗ aspace)
     F = TensorMap(ones, aspace ⊗ pspace, pspace)

     Fdag, F
end

# hopping term, FFdag
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
const FFdag = let
     aspace = Rep[U₁](1 => 1)
     iso = isometry(aspace, flip(aspace))
     @tensor F[a; c d] := FdagF[1]'[a, b, c] * iso'[d, b]
     @tensor Fdag[d a; c] := FdagF[2]'[a, b, c] * iso[b, d]

     F, Fdag
end

const ΔdagΔ = let
     A = FdagF[1]
     aspace = Rep[U₁](1 => 1)
     aspace2 = Rep[U₁](2 => 1)
     iso = isometry(aspace ⊗ aspace, aspace2)
     @tensor B[d a; b e] := A[a b c] * iso[c d e]
     C = permute(B', ((2, 1), (4, 3)))
     D = permute(A', ((2, 1), (3,)))
     A, B, C, D
end

end

"""
     const U1SpinlessFermion = U₁SpinlessFermion
"""
const U1SpinlessFermion = U₁SpinlessFermion