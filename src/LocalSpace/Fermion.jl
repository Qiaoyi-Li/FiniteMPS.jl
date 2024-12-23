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

     SSS::NTuple{3, TensorMap}
Three operators of chiral operator `imag(S⋅(S×S))`. Rank = `(3, 4, 3)`.
SSS = imag(S⋅(S×S)) = -im * S⋅(S×S)  -->  S⋅(S×S) = im * SSS
NOTICE: The chiral operator `S⋅(S×S)` is a pure imaginary operator under the current basis.
Thus define `SSS` as the imaginary part of S⋅(S×S) to reduce the computational overhead.

     FdagF::NTuple{2, TensorMap}
Two rank-`3` operators of hopping `c↑^dag c↑ + c↓^dag c↓`. 

     FFdag::NTuple{2, TensorMap}
Two rank-`3` operators of hopping `c↑ c↑^dag + c↓ c↓^dag`. 

     ΔₛdagΔₛ::NTuple{4, TensorMap}
Four operators of singlet pairing correlation `Δₛ^dagΔₛ`, where `Δₛ = (c↓c↑ - c↑c↓)/√2`. Rank = `(3, 4, 4, 3)`.

     ΔₜdagΔₜ::NTuple{4, TensorMap}
Four operators of triplet pairing correlation `Δₜ^dag⋅Δₜ`, where `Δₜ` is the triplet pairing operator that carries `2` charge and `1` spin quantum numbers. Rank = `(3, 4, 4, 3)`.

     Δₛ::NTuple{2, TensorMap}
     Δₛdag::NTuple{2, TensorMap}
Singlet pairing operators `Δₛ` and `Δₛ^dag`. Rank = `(4, 3)`. Note the first operator has nontrivial left bond index.

     CpCm::NTuple{2, TensorMap}
Two rank-`3` operators of `C+C-` correlation where `C+ = c↑^dag c↓^dag` and `C- = C+^dag = c↓c↑`. Note both operators are bosonic.
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
     C = permute(B', ((2, 1), (4, 3)))
     D = permute(A', ((2, 1), (3,)))
     A, B, C, D
end

# triplet pairing correlation
const ΔₜdagΔₜ = let
     A = FdagF[1]
     aspace = Rep[U₁×SU₂]((1, 1 / 2) => 1)
     aspace2 = Rep[U₁×SU₂]((2, 1) => 1)
     iso = isometry(aspace ⊗ aspace, aspace2)
     @tensor B[d a; b e] := A[a b c] * iso[c d e]
     C = permute(B', ((2, 1), (4, 3)))
     # -1 here as Δₜ is anti-symmetric with site indices
     D = -permute(A', ((2, 1), (3,)))
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

# C+C- correlation
const CpCm = let Fdag = FdagF[1]
     aspace = Rep[U₁×SU₂]((1, 1 / 2) => 1)
     aspace2 = Rep[U₁×SU₂]((2, 0) => 1)
     iso = isometry(aspace ⊗ aspace, aspace2)
     @tensor Cp[a; d f] := Fdag[a b c] * Fdag[b d e] * iso[c e f]
     Cm = permute(Cp', ((2, 1), (3,)))
     Cp, Cm
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
     SR = permute(SL', ((2, 1), (3,)))
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
     C = permute(B', ((1,), (3, 2)))
     D = permute(A', ((2, 1), (3,)))
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

     ΔdagΔ₊₊::NTuple{4, TensorMap}
     ΔdagΔ₋₋::NTuple{4, TensorMap}
     ΔdagΔ₊₋::NTuple{4, TensorMap}
Rank-`4` operators of pairing correlation. Note `ΔdagΔ₊₋` means `c↑^dag c↓^dag c↓ c↑`.

     EdagE₊::NTuple{4, TensorMap}
     EdagE₋::NTuple{4, TensorMap}
Rank-`4` operators of triplet exciton correlation. Note `EdagE₊` means `c↑^dag c↓ c↑ c↓^dag` so `(i, j, i, j)` gives the correlation of the same pair.  
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

const EdagE₊ = let
     A, C = FdagF₊
     B, D = FFdag₋
     aspace2 = Rep[U₁×U₁]((0, 1) => 1)
     iso = isometry(domain(A)[2] ⊗ domain(B)[2], aspace2)
     @tensor B[d a; b e] := B[a b c] * iso[d c e]
     @tensor C[a d; e c] := iso'[a b c] * C[b d e]
     A, B, C, D
end

const EdagE₋ = let
     A, C = FdagF₋
     B, D = FFdag₊
     aspace2 = Rep[U₁×U₁]((0, -1) => 1)
     iso = isometry(domain(A)[2] ⊗ domain(B)[2], aspace2)
     @tensor B[d a; b e] := B[a b c] * iso[d c e]
     @tensor C[a d; e c] := iso'[a b c] * C[b d e]
     A, B, C, D
end

end

"""
     const U1U1Fermion = U₁U₁Fermion
"""
const U1U1Fermion = U₁U₁Fermion

