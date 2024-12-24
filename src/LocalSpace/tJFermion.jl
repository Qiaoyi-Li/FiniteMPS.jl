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

# chiral operator imag(S⋅(S×S))
const SSS = let
    aspace = Rep[U₁×SU₂]((0, 1) => 1)

    SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace)
    SM = TensorMap(zeros, Float64, aspace ⊗ pspace, pspace ⊗ aspace)
    block(SM, Irrep[U₁×SU₂](0, 1/2)) .= 3/4
    block(SM, Irrep[U₁×SU₂](0, 3/2)) .= 3/8
    SR = TensorMap(ones, Float64, aspace ⊗ pspace, pspace)

    SL, SM, SR
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

"""
     module U₁U₁tJFermion

Prepare some commonly used objects for U₁×U₁ `tJ` fermions, i.e. local `d = 3` Hilbert space without double occupancy. 
     
Behaviors of all operators are the same as `U₁U₁Fermion` up to the projection, details please see `U₁U₁Fermion`. 
"""
module U₁U₁tJFermion

using TensorKit

const pspace = Rep[U₁×U₁]((-1, 0) => 1, (0, -1 // 2) => 1, (0, 1 // 2) => 1)

const Z = let
     Z = TensorMap(ones, pspace, pspace)
     block(Z, Irrep[U₁×U₁](0, 1 // 2)) .= -1
     block(Z, Irrep[U₁×U₁](0, -1 // 2)) .= -1
     Z
end

const n₊ = let
     n₊ = TensorMap(zeros, pspace, pspace)
     block(n₊, Irrep[U₁×U₁](0, 1 / 2)) .= 1
     n₊
end

const n₋ = let
     n₋ = TensorMap(zeros, pspace, pspace)
     block(n₋, Irrep[U₁×U₁](0, -1 / 2)) .= 1
     n₋
end

const n = n₊ + n₋

const Sz = (n₊ - n₋) / 2

# S+ S- interaction
# convention: S⋅S = SzSz + (S₊₋ + S₋₊)/2
const S₊₋ = let
     aspace = Rep[U₁×U₁]((0, 1) => 1)
     S₊ = TensorMap(ones, pspace, pspace ⊗ aspace)
     S₋ = TensorMap(ones, aspace ⊗ pspace, pspace)
     S₊, S₋
end

const S₋₊ = let
     aspace = Rep[U₁×U₁]((0, 1) => 1)
     iso = isometry(aspace, flip(aspace))
     @tensor S₋[a; c d] := S₊₋[1]'[a, b, c] * iso'[d, b]
     @tensor S₊[d a; c] := S₊₋[2]'[a, b, c] * iso[b, d]
     S₋, S₊
end

# hopping term, FdagF₊ = c↑^dag c↑
const FdagF₊ = let
     aspace = Rep[U₁×U₁]((1, 1 // 2) => 1)
     Fdag₊ = TensorMap(ones, pspace, pspace ⊗ aspace)
     F₊ = TensorMap(ones, aspace ⊗ pspace, pspace)
     Fdag₊, F₊
end
const FdagF₋ = let
     # note c↓^dag|↑⟩ = -|↑↓⟩, c↓|↑↓⟩ = -|↑⟩  
     aspace = Rep[U₁×U₁]((1, -1 // 2) => 1)
     Fdag₋ = TensorMap(ones, pspace, pspace ⊗ aspace)
     block(Fdag₋, Irrep[U₁×U₁](1, 0)) .= -1
     F₋ = TensorMap(ones, aspace ⊗ pspace, pspace)
     block(F₋, Irrep[U₁×U₁](1, 0)) .= -1
     Fdag₋, F₋
end
const FFdag₊ = let
     aspace = Rep[U₁×U₁]((1, 1 // 2) => 1)
     iso = isometry(aspace, flip(aspace))
     @tensor F₊[a; c d] := FdagF₊[1]'[a, b, c] * iso'[d, b]
     @tensor Fdag₊[d a; c] := FdagF₊[2]'[a, b, c] * iso[b, d]
     F₊, Fdag₊
end
const FFdag₋ = let
     aspace = Rep[U₁×U₁]((1, -1 // 2) => 1)
     iso = isometry(aspace, flip(aspace))
     @tensor F₋[a; c d] := FdagF₋[1]'[a, b, c] * iso'[d, b]
     @tensor Fdag₋[d a; c] := FdagF₋[2]'[a, b, c] * iso[b, d]
     F₋, Fdag₋
end

const ΔdagΔ₊₊ = let
     A = FdagF₊[1]
     aspace = Rep[U₁×U₁]((1, 1 / 2) => 1)
     aspace2 = Rep[U₁×U₁]((2, 1) => 1)
     iso = isometry(aspace ⊗ aspace, aspace2)
     @tensor B[d a; b e] := A[a b c] * iso[c d e]
     C = permute(B', ((2, 1), (4, 3)))
     D = permute(A', ((2, 1), (3,)))
     A, B, C, D
end
const ΔdagΔ₋₋ = let
     A = FdagF₋[1]
     aspace = Rep[U₁×U₁]((1, -1 / 2) => 1)
     aspace2 = Rep[U₁×U₁]((2, -1) => 1)
     iso = isometry(aspace ⊗ aspace, aspace2)
     @tensor B[d a; b e] := A[a b c] * iso[c d e]
     C = permute(B', ((2, 1), (4, 3)))
     D = permute(A', ((2, 1), (3,)))
     A, B, C, D
end
const ΔdagΔ₊₋ = let
     A = FdagF₊[1]
     B = FdagF₋[1]
     aspace_A = Rep[U₁×U₁]((1, 1 / 2) => 1)
     aspace_B = Rep[U₁×U₁]((1, -1 / 2) => 1)
     aspace2 = Rep[U₁×U₁]((2, 0) => 1)
     iso = isometry(aspace_A ⊗ aspace_B, aspace2)
     @tensor B[d a; b e] := B[a b c] * iso[d c e]
     C = permute(B', ((2, 1), (4, 3)))
     D = permute(A', ((2, 1), (3,)))
     A, B, C, D
end

end

"""
     const U1U1tJFermion = U₁U₁tJFermion
"""
const U1U1tJFermion = U₁U₁tJFermion


module ℤ₂SU₂tJFermion

using TensorKit

const pspace = Rep[ℤ₂×SU₂]((0, 0) => 1, (1, 1 // 2) => 1)

const Z = let
     Z = isometry(pspace, pspace)
     block(Z, Irrep[ℤ₂×SU₂](1, 1 // 2)) .= -1
     Z
end

const n = let
     n = isometry(pspace, pspace)
     block(n, Irrep[ℤ₂×SU₂](0, 0))[1, 1] = 0
     n
end

# S⋅S interaction
const SS = let
     aspace = Rep[ℤ₂×SU₂]((0, 1) => 1)
     SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2
     SR = permute(SL', ((2, 1), (3,)))
     SL, SR
end

# hopping term, FdagF
const FdagF = let
     aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
     Fdag = TensorMap(zeros, pspace, pspace ⊗ aspace)
     block(Fdag, Irrep[ℤ₂×SU₂](1, 1 // 2))[1, 1] = 1
     F = TensorMap(zeros, aspace ⊗ pspace, pspace)
     block(F, Irrep[ℤ₂×SU₂](1, 1 // 2))[1, 1] = 1

     Fdag, F
end

# hopping term, FFdag
# warning: each hopping term == Fᵢ^dag Fⱼ - Fᵢ Fⱼ^dag
const FFdag = let

     aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
     iso = isometry(aspace, flip(aspace))
     @tensor F[a; c d] := FdagF[1]'[a, b, c] * iso'[d, b]
     @tensor Fdag[d a; c] := FdagF[2]'[a, b, c] * iso[b, d]

     F, Fdag
end

# singlet pairing correlation Δᵢⱼ^dag Δₖₗ
const ΔₛdagΔₛ = let
     A = FdagF[1]
     aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
     iso = isometry(aspace, flip(aspace)) / sqrt(2)
     @tensor B[d a; b] := A[a b c] * iso[c d]
     C = permute(B', ((1,), (3, 2)))
     D = permute(A', ((2, 1), (3,)))
     A, B, C, D
end

# singlet pairing operator
const Δₛ = (ΔₛdagΔₛ[3], ΔₛdagΔₛ[4])

const Δₛdag = (ΔₛdagΔₛ[1], ΔₛdagΔₛ[2])

end

"""
     const Z2SU2tJFermion = ℤ₂SU₂tJFermion
"""
const Z2SU2tJFermion = ℤ₂SU₂tJFermion
