"""
     module U₁SU₂tJFermion

Prepare some commonly used objects for U₁×SU₂ `tJ` fermions, i.e. local `d = 3` Hilbert space without double occupancy. 
     
Behaviors of all operators are the same as `U₁SU₂Fermion` up to the projection, details please see `U₁SU₂Fermion`. 
"""
module U₁SU₂tJFermion

using TensorKit

const pspace = Rep[U₁×SU₂]((-1, 0) => 1, (0, 1 // 2) => 1)
const Z = let
     Z = TensorMap(ones, pspace, pspace)
     block(Z, Irrep[U₁×SU₂](0, 1 / 2)) .= -1
     Z
end

const n = let
     n = TensorMap(ones, pspace, pspace)
     block(n, Irrep[U₁×SU₂](-1, 0)) .= 0
     n
end

# S⋅S interaction
const SS = let
     aspace = Rep[U₁×SU₂]((0, 1) => 1)
     SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2

     SR = permute(SL', ((2, 1), (3,)))
     SL, SR
end

# hopping term, FdagF
const FdagF = let
     aspace = Rep[U₁×SU₂]((1, 1 / 2) => 1)
     Fdag = TensorMap(ones, pspace, pspace ⊗ aspace)
     F = TensorMap(ones, aspace ⊗ pspace, pspace)

     Fdag, F
end

# hopping term, FFdag
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
const FFdag = let

     aspace = Rep[U₁×SU₂]((1, 1 / 2) => 1)
     iso = isometry(aspace, flip(aspace))
     @tensor F[a; c d] := FdagF[1]'[a, b, c] * iso'[d, b]
     @tensor Fdag[d a; c] := FdagF[2]'[a, b, c] * iso[b, d]

     F, Fdag
end

# singlet pairing Δᵢⱼ^dag Δₖₗ
const ΔₛdagΔₛ = let
     A = FdagF[1]
     aspace = Rep[U₁×SU₂]((1, 1 / 2) => 1)
     aspace2 = Rep[U₁×SU₂]((2, 0) => 1)
     iso = isometry(aspace ⊗ aspace, aspace2)
     @tensor B[d a; b e] := A[a b c] * iso[c d e]
     C = permute(B', ((2, 1), (4, 3)))
     D = permute(A', ((2, 1), (3,)))
     A, B, C, D
end

# singlet pairing operator
const Δₛ = (ΔₛdagΔₛ[3], ΔₛdagΔₛ[4])

const Δₛdag = let (A, B) = Δₛ
     Adag = permute(A', ((3, 1), (4, 2)))
     iso = isometry(flip(codomain(Adag)[1]), codomain(Adag)[1])
     @tensor Adag[a c; d e] := iso[a b] * Adag[b c; d e]
     Bdag = permute(B', ((2, 1), (3,)))
     Adag, -Bdag
end


end

"""
     const U1SU2tJFermion = U₁SU₂tJFermion
"""
const U1SU2tJFermion = U₁SU₂tJFermion
