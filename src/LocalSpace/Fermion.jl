"""
     module U₁SU₂Fermion  

Prepare some commonly used objects for U₁×SU₂ fermions.

Nothing is exported, please call `U₁SU₂Fermion.xxx` to use `xxx`. 

# Fields
     pspace::VectorSpace  
Local `d = 4` Hilbert space.

     Z::TensorMap
Rank-`2` fermion parity operator `Z = (-1)^n`.

     n::TensorMap
Rank-`2` particle number operator `n = n↑ + n↓`.

     nd::TensorMap
Rank-`2` double occupancy operator `nd = n↑n↓`.

     SS::NTuple{2, TensorMap}
Two rank-`3` operators of Heisenberg `S⋅S` interaction.  

     FdagF::NTuple{2, TensorMap}
Two rank-`3` operators of hopping `c↑^dag c↑ + c↓^dag c↓`. 

     FFdag::NTuple{2, TensorMap}
Two rank-`3` operators of hopping `c↑ c↑^dag + c↓ c↓^dag`. 

     ΔₛdagΔₛ::NTuple{4, TensorMap}
Four operators of singlet pairing correlation `Δₛ^dagΔₛ`, where `Δₛ = (c↓c↑ - c↑c↓)/√2`. Rank = `(3, 4, 4, 3)`.

     Δₛ::NTuple{2, TensorMap}
     Δₛdag::NTuple{2, TensorMap}
Singlet pairing operators `Δₛ` and `Δₛ^dag`. Rank = `(4, 3)`. Note the first operator has nontrivial left bond index.
"""
module U₁SU₂Fermion

using TensorKit

const pspace = Rep[U₁×SU₂]((-1, 0) => 1, (0, 1 // 2) => 1, (1, 0) => 1)
const Z = let
     Z = TensorMap(ones, pspace, pspace)
     block(Z, Irrep[U₁×SU₂](0, 1 / 2)) .= -1
     Z
end

const n = let
     n = TensorMap(ones, pspace, pspace)
     block(n, Irrep[U₁×SU₂](1, 0)) .= 2
     block(n, Irrep[U₁×SU₂](-1, 0)) .= 0
     n
end

const nd = let
     nd = TensorMap(zeros, pspace, pspace)
     block(nd, Irrep[U₁×SU₂](1, 0)) .= 1
     nd
end

# S⋅S interaction
const SS = let
     aspace = Rep[U₁×SU₂]((0, 1) => 1)
     SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2

     SR = permute(SL', (2, 1), (3,))
     SL, SR
end

# hopping term, FdagF
const FdagF = let
     aspace = Rep[U₁×SU₂]((1, 1 / 2) => 1)
     Fdag = TensorMap(ones, pspace, pspace ⊗ aspace)
     block(Fdag, Irrep[U₁×SU₂](1, 0)) .= -sqrt(2)
     F = TensorMap(ones, aspace ⊗ pspace, pspace)
     block(F, Irrep[U₁×SU₂](1, 0)) .= sqrt(2)

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

# singlet pairing correlation Δᵢⱼ^dag Δₖₗ
const ΔₛdagΔₛ = let
     A = FdagF[1]
     aspace = Rep[U₁×SU₂]((1, 1 / 2) => 1)
     aspace2 = Rep[U₁×SU₂]((2, 0) => 1)
     iso = isometry(aspace ⊗ aspace, aspace2)
     @tensor B[d a; b e] := A[a b c] * iso[c d e]
     C = permute(B', (2, 1), (4, 3))
     D = permute(A', (2, 1), (3,))
     A, B, C, D
end

# singlet pairing operator
const Δₛ = (ΔₛdagΔₛ[3], ΔₛdagΔₛ[4])

const Δₛdag = let (A, B) = Δₛ
     Adag = permute(A', (3, 1), (4, 2))
     iso = isometry(flip(codomain(Adag)[1]), codomain(Adag)[1])
     @tensor Adag[a c; d e] := iso[a b] * Adag[b c; d e]
     Bdag = permute(B', (2, 1), (3,))
     Adag, -Bdag
end

end

"""
     const U1SU2Fermion = U₁SU₂Fermion
"""
const U1SU2Fermion = U₁SU₂Fermion

"""
     module ℤ₂SU₂Fermion

Prepare some commonly used objects for ℤ₂×SU₂ fermions. Basis convention in `(0, 0)` sector is `{|0⟩, |↑↓⟩}`.

Each operator has the same name and behavior as `U₁SU₂Fermion`, details please see `U₁SU₂Fermion`.
"""
module ℤ₂SU₂Fermion

using TensorKit

const pspace = Rep[ℤ₂×SU₂]((0, 0) => 2, (1, 1 // 2) => 1)

const Z = let
     Z = isometry(pspace, pspace)
     block(Z, Irrep[ℤ₂×SU₂](1, 1 // 2)) .= -1
     Z
end

const n = let
     n = isometry(pspace, pspace)
     block(n, Irrep[ℤ₂×SU₂](0, 0))[1, 1] = 0
     block(n, Irrep[ℤ₂×SU₂](0, 0))[2, 2] = 2
     n
end

const nd = let
     nd = TensorMap(zeros, pspace, pspace)
     block(nd, Irrep[ℤ₂×SU₂](0, 0))[2, 2] = 1
     nd
end

# S⋅S interaction
const SS = let
     aspace = Rep[ℤ₂×SU₂]((0, 1) => 1)
     SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2
     SR = permute(SL', (2, 1), (3,))
     SL, SR
end

# hopping term, FdagF
const FdagF = let
     aspace = Rep[ℤ₂×SU₂]((1, 1 // 2) => 1)
     Fdag = TensorMap(zeros, pspace, pspace ⊗ aspace)
     block(Fdag, Irrep[ℤ₂×SU₂](1, 1 // 2))[1, 1] = 1
     block(Fdag, Irrep[ℤ₂×SU₂](0, 0))[2, 1] = sqrt(2)
     F = TensorMap(zeros, aspace ⊗ pspace, pspace)
     block(F, Irrep[ℤ₂×SU₂](1, 1 // 2))[1, 1] = 1
     block(F, Irrep[ℤ₂×SU₂](0, 0))[1, 2] = -sqrt(2)

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
     C = permute(B', (1,), (3, 2))
     D = permute(A', (2, 1), (3,))
     A, B, C, D
end

# singlet pairing operator
const Δₛ = (ΔₛdagΔₛ[3], ΔₛdagΔₛ[4])

const Δₛdag = (ΔₛdagΔₛ[1], ΔₛdagΔₛ[2])

end

"""
     const Z2SU2Fermion = ℤ₂SU₂Fermion
"""
const Z2SU2Fermion = ℤ₂SU₂Fermion

"""
     module U₁U₁Fermion

Prepare the local space of `d = 4` spin-1/2 fermions with `U₁` charge and `U₁` spin symmetry.

# Fields
     pspace::VectorSpace
Local `d = 4` Hilbert space.

     Z::TensorMap
Rank-`2` fermion parity operator `Z = (-1)^n`.

     n₊::TensorMap
     n₋::TensorMap
     n::TensorMap
Rank-`2` particle number operators. `₊` and `₋` denote spin up and down as `↑` and `↓` cannot be used in variable names.

     nd::TensorMap
Rank-`2` double occupancy operator `nd = n↑n↓`.

     Sz::TensorMap
Rank-`2` spin-z operator `Sz = (n↑ - n↓)/2`.

     S₊₋::NTuple{2, TensorMap}
     S₋₊::NTuple{2, TensorMap}
Two rank-`3` operators of `S₊₋` and `S₋₊` interaction. Note Heisenberg `S⋅S = SzSz + (S₊₋ + S₋₊)/2`.

     FdagF₊::NTuple{2, TensorMap}
     FdagF₋::NTuple{2, TensorMap}
Two rank-`3` operators of hopping `c↑^dag c↑` and `c↓^dag c↓`.

     FFdag₊::NTuple{2, TensorMap}
     FFdag₋::NTuple{2, TensorMap}
Two rank-`3` operators of hopping `c↑ c↑^dag` and `c↓ c↓^dag`.
"""
module U₁U₁Fermion

using TensorKit

const pspace = Rep[U₁×U₁]((-1, 0) => 1, (0, -1 // 2) => 1, (0, 1 // 2) => 1, (1, 0) => 1)

const Z = let
     Z = TensorMap(ones, pspace, pspace)
     block(Z, Irrep[U₁×U₁](0, 1 // 2)) .= -1
     block(Z, Irrep[U₁×U₁](0, -1 // 2)) .= -1
     Z
end

const n₊ = let
     n₊ = TensorMap(zeros, pspace, pspace)
     block(n₊, Irrep[U₁×U₁](1, 0)) .= 1
     block(n₊, Irrep[U₁×U₁](0, 1 / 2)) .= 1
     n₊
end

const n₋ = let
     n₋ = TensorMap(zeros, pspace, pspace)
     block(n₋, Irrep[U₁×U₁](1, 0)) .= 1
     block(n₋, Irrep[U₁×U₁](0, -1 / 2)) .= 1
     n₋
end


const n = n₊ + n₋

const nd = n₊ * n₋

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

# TODO pairing Δ↑↓^dag, ...


end

"""
     const U1U1Fermion = U₁U₁Fermion
"""
const U1U1Fermion = U₁U₁Fermion

