"""
    module U₁SU₂tJBoson

U(1) × SU(2) local space for two-component hard-core bosons with total
occupation n ≤ 1: |0⟩, |↑⟩, |↓⟩. The U(1) charge is q = n - 1, following
FiniteMPS.U₁SU₂tJFermion; the empty state has (q, S) = (-1, 0).

Operators are TensorKit tensors or tuples with symmetry-carrying auxiliary
legs. Nothing is exported; use U1SU2tJBoson.pspace, U1SU2tJBoson.BdagB, etc.
For FiniteMPS.addIntr! / addObs!, mark EVERY endpoint `fermionic=false`
and omit the `Z` keyword. No Jordan–Wigner parity string is defined here.

For each distinct-site bond (i, j), the hopping Hamiltonian is
    -t * BdagB(i,j) - conj(t) * BdagB(j,i).
Equivalently the second term is -conj(t) * BBdag(i,j), with no fermionic
relative minus sign. SS is Sᵢ⋅Sⱼ alone; any density interaction is separate.
"""
module U₁SU₂tJBoson

using TensorKit

"""Three-state space with U(1) charge q = n - 1 and spin S = 0 or 1/2."""
const pspace = Rep[U₁×SU₂]((-1, 0) => 1, (0, 1 // 2) => 1)

"""Total particle number n = n↑ + n↓, a projector onto single occupancy."""
const n = let
    n = ones(pspace, pspace)
    block(n, Irrep[U₁×SU₂](-1, 0)) .= 0
    n
end

"""Two endpoints for Sᵢ⋅Sⱼ; does not include -nᵢnⱼ/4. Sᵢ² = 3nᵢ/4."""
const SS = let
    aspace = Rep[U₁×SU₂]((0, 1) => 1)
    SL = ones(pspace, pspace ⊗ aspace) * sqrt(3) / 2
    SR = permute(SL', ((2, 1), (3,)))
    SL, SR
end

"""
Three endpoints for -im * Sᵢ⋅(Sⱼ×Sₖ), using FiniteMPS's real-tensor
convention. Multiply by im to obtain the physical scalar chirality.
"""
const SSS = let
    aspace = Rep[U₁×SU₂]((0, 1) => 1)
    SL = ones(pspace, pspace ⊗ aspace)
    SM = zeros(aspace ⊗ pspace, pspace ⊗ aspace)
    block(SM, Irrep[U₁×SU₂](0, 1 // 2)) .= 3 / 4
    block(SM, Irrep[U₁×SU₂](0, 3 // 2)) .= 3 / 8
    SR = ones(aspace ⊗ pspace, pspace)
    SL, SM, SR
end

"""
Two endpoints for ∑σ b†ᵢσ bⱼσ, where b†σ = |σ⟩⟨0|.
Their ranks are (3, 3); the auxiliary leg carries (q, S) = (1, 1/2).
"""
const BdagB = let
    aspace = Rep[U₁×SU₂]((1, 1 // 2) => 1)
    Bdag = ones(pspace, pspace ⊗ aspace)
    B = ones(aspace ⊗ pspace, pspace)
    Bdag, B
end

"""
Two endpoints for ∑σ bᵢσ b†ⱼσ. For i ≠ j this is BdagB(j,i).
For coincident sites it equals 2(1-nᵢ), whereas BdagB(i,i) = nᵢ.
"""
const BBdag = let
    aspace = Rep[U₁×SU₂]((1, 1 // 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor B[a; c d] := BdagB[1]'[a, b, c] * iso'[d, b]
    @tensor Bdag[d a; c] := BdagB[2]'[a, b, c] * iso[b, d]
    B, Bdag
end

"""
Four endpoints for Δₛ†(i,j) Δₛ(k,l), in that site order, with
    Δₛ(i,j) = (bᵢ↓ bⱼ↑ - bᵢ↑ bⱼ↓) / √2.
For bosons Δₛ(j,i) = -Δₛ(i,j). All four endpoints are bosonic.
The middle auxiliary leg carries (q, S) = (2, 0).
"""
const ΔₛdagΔₛ = let
    A = BdagB[1]
    aspace = Rep[U₁×SU₂]((1, 1 // 2) => 1)
    pairspace = Rep[U₁×SU₂]((2, 0) => 1)
    iso = isometry(aspace ⊗ aspace, pairspace)
    @tensor B[d a; b e] := A[a b c] * iso[c d e]
    C = permute(B', ((2, 1), (4, 3)))
    # The antisymmetric singlet contraction needs this phase for the
    # stated annihilator order (k,l), without fermionic reordering.
    D = -permute(A', ((2, 1), (3,)))
    A, B, C, D
end

"""
Four endpoints for ∑ₘ Δₜ,ₘ†(i,j) Δₜ,ₘ(k,l), in that site order, where
    Δₜ,+1(i,j) = bᵢ↑ bⱼ↑,
    Δₜ, 0(i,j) = (bᵢ↑ bⱼ↓ + bᵢ↓ bⱼ↑) / √2,
    Δₜ,-1(i,j) = bᵢ↓ bⱼ↓.
For bosons Δₜ,ₘ(j,i) = Δₜ,ₘ(i,j). The middle auxiliary leg carries
(q, S) = (2, 1). There is no extra factor of three in this definition.
"""
const ΔₜdagΔₜ = let
    A = BdagB[1]
    aspace = Rep[U₁×SU₂]((1, 1 // 2) => 1)
    pairspace = Rep[U₁×SU₂]((2, 1) => 1)
    iso = isometry(aspace ⊗ aspace, pairspace)
    @tensor B[d a; b e] := A[a b c] * iso[c d e]
    C = permute(B', ((2, 1), (4, 3)))
    # Bosonic triplet pairs are symmetric under exchange of the sites.
    D = permute(A', ((2, 1), (3,)))
    A, B, C, D
end

"""
Two endpoints for Δₛ(i,j) = (bᵢ↓ bⱼ↑ - bᵢ↑ bⱼ↓)/√2, which lowers N by two.
For `calObs!` between compatible number sectors, supply an explicit `El`
that contracts the open left auxiliary leg with charge (2, 0).
Use `ΔₛdagΔₛ` for number-conserving pair correlations.
"""
const Δₛ = (ΔₛdagΔₛ[3], ΔₛdagΔₛ[4])

"""
Two endpoints for [Δₛ(i,j)]† at the same ordered sites (i,j), which raises N by two.
For `calObs!` between compatible number sectors, supply an explicit `El`
that contracts the open left auxiliary leg with charge (-2, 0).
"""
const Δₛdag = let (A, B) = Δₛ
    Adag = permute(A', ((3, 1), (4, 2)))
    iso = isometry(flip(codomain(Adag)[1]), codomain(Adag)[1])
    @tensor Adag[a c; d e] := iso[a b] * Adag[b c; d e]
    Bdag = permute(B', ((2, 1), (3,)))
    Adag, Bdag
end

end # module U₁SU₂tJBoson

"""ASCII alias of U₁SU₂tJBoson."""
const U1SU2tJBoson = U₁SU₂tJBoson

"""
    module U₁U₁tJBoson

U(1) × U(1) local space for two-component hard-core bosons with total
occupation n ≤ 1. The charges are (q, m) = (n - 1, Sz), so the states
|0⟩, |↑⟩, |↓⟩ carry (-1, 0), (0, 1/2), (0, -1/2), respectively.

Use n₊, n₋, Sz and S₊₋/S₋₊ for spin-resolved observables and exchange:
Sᵢ⋅Sⱼ = Szᵢ Szⱼ + (S₊₋(i,j) + S₋₊(i,j))/2.
BdagB₊/BdagB₋ are b†ᵢ↑bⱼ↑/b†ᵢ↓bⱼ↓; BBdag₊/BBdag₋ are the reverse
products. For a distinct-site bond, each spin's hopping is
-tσ * BdagBσ(i,j) - conj(tσ) * BBdagσ(i,j).

All endpoints passed to FiniteMPS.addIntr! / addObs! must have
fermionic=false, with no Z keyword. Nothing is exported.
"""
module U₁U₁tJBoson

using TensorKit

const pspace = Rep[U₁×U₁]((-1, 0) => 1, (0, -1 // 2) => 1, (0, 1 // 2) => 1)

"""Projector onto |↑⟩."""
const n₊ = let
    n₊ = zeros(pspace, pspace)
    block(n₊, Irrep[U₁×U₁](0, 1 // 2)) .= 1
    n₊
end

"""Projector onto |↓⟩."""
const n₋ = let
    n₋ = zeros(pspace, pspace)
    block(n₋, Irrep[U₁×U₁](0, -1 // 2)) .= 1
    n₋
end

const n = n₊ + n₋

const Sz = (n₊ - n₋) / 2

"""Two endpoints for S⁺ᵢ S⁻ⱼ."""
const S₊₋ = let
    aspace = Rep[U₁×U₁]((0, 1) => 1)
    S₊ = ones(pspace, pspace ⊗ aspace)
    S₋ = ones(aspace ⊗ pspace, pspace)
    S₊, S₋
end

"""Two endpoints for S⁻ᵢ S⁺ⱼ."""
const S₋₊ = let
    aspace = Rep[U₁×U₁]((0, 1) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor S₋[a; c d] := S₊₋[1]'[a, b, c] * iso'[d, b]
    @tensor S₊[d a; c] := S₊₋[2]'[a, b, c] * iso[b, d]
    S₋, S₊
end

"""Two endpoints for b†ᵢ↑ bⱼ↑, with auxiliary charge (1, 1/2)."""
const BdagB₊ = let
    aspace = Rep[U₁×U₁]((1, 1 // 2) => 1)
    Bdag₊ = ones(pspace, pspace ⊗ aspace)
    B₊ = ones(aspace ⊗ pspace, pspace)
    Bdag₊, B₊
end

"""Two endpoints for b†ᵢ↓ bⱼ↓, with auxiliary charge (1, -1/2)."""
const BdagB₋ = let
    aspace = Rep[U₁×U₁]((1, -1 // 2) => 1)
    Bdag₋ = ones(pspace, pspace ⊗ aspace)
    B₋ = ones(aspace ⊗ pspace, pspace)
    Bdag₋, B₋
end

"""Two endpoints for bᵢ↑ b†ⱼ↑; no fermionic relative minus sign."""
const BBdag₊ = let
    aspace = Rep[U₁×U₁]((1, 1 // 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor B₊[a; c d] := BdagB₊[1]'[a, b, c] * iso'[d, b]
    @tensor Bdag₊[d a; c] := BdagB₊[2]'[a, b, c] * iso[b, d]
    B₊, Bdag₊
end

"""Two endpoints for bᵢ↓ b†ⱼ↓; no fermionic relative minus sign."""
const BBdag₋ = let
    aspace = Rep[U₁×U₁]((1, -1 // 2) => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor B₋[a; c d] := BdagB₋[1]'[a, b, c] * iso'[d, b]
    @tensor Bdag₋[d a; c] := BdagB₋[2]'[a, b, c] * iso[b, d]
    B₋, Bdag₋
end

# Build [bᵢα bⱼβ]† [bₖα bₗβ] in the endpoint order (i,j,k,l).
# For mixed spins the annihilators must retain (α,β), not (β,α).
function _pair_correlation(Aα, Aβ, mα, mβ)
    aspaceα = Rep[U₁×U₁]((1, mα) => 1)
    aspaceβ = Rep[U₁×U₁]((1, mβ) => 1)
    pairspace = Rep[U₁×U₁]((2, mα + mβ) => 1)
    iso = isometry(aspaceα ⊗ aspaceβ, pairspace)
    @tensor B[d a; b e] := Aβ[a b c] * iso[d c e]
    @tensor Bswap[d a; b e] := Aα[a b c] * iso[c d e]
    C = permute(Bswap', ((2, 1), (4, 3)))
    D = permute(Aβ', ((2, 1), (3,)))
    Aα, B, C, D
end

"""Four endpoints for [bᵢ↑ bⱼ↑]† [bₖ↑ bₗ↑], with no normalization factor."""
const ΔdagΔ₊₊ = _pair_correlation(BdagB₊[1], BdagB₊[1], 1 // 2, 1 // 2)

"""Four endpoints for [bᵢ↓ bⱼ↓]† [bₖ↓ bₗ↓], with no normalization factor."""
const ΔdagΔ₋₋ = _pair_correlation(BdagB₋[1], BdagB₋[1], -1 // 2, -1 // 2)

"""
Four endpoints for [bᵢ↑ bⱼ↓]† [bₖ↑ bₗ↓], in site order (i,j,k,l).
This is a spin-resolved channel, not the singlet or the m=0 triplet.
"""
const ΔdagΔ₊₋ = _pair_correlation(BdagB₊[1], BdagB₋[1], 1 // 2, -1 // 2)

end # module U₁U₁tJBoson

"""ASCII alias of U₁U₁tJBoson."""
const U1U1tJBoson = U₁U₁tJBoson

"""
    module SU₂tJBoson

SU(2) local space for two-component hard-core bosons with total occupation
n ≤ 1: |0⟩ is a spin-0 singlet and |↑⟩, |↓⟩ form a spin-1/2 doublet.
Only total spin is a symmetry label; no U(1) particle-number charge is kept.
Particle-number parity is already fixed by (-1)^N = (-1)^(2S), so a separate
Z₂ label would be redundant. All tensors use ordinary bosonic statistics.

The hopping convention is -t * BdagB(i,j) - conj(t) * BBdag(i,j) for i ≠ j.
SS is Sᵢ⋅Sⱼ without a density term. All endpoints passed to
FiniteMPS.addIntr! / addObs! must have fermionic=false, with no Z keyword.
Nothing is exported.
"""
module SU₂tJBoson

using TensorKit

const pspace = Rep[SU₂](0 => 1, 1 // 2 => 1)

"""Single-occupancy projector; particle number is not a symmetry label."""
const n = let
    n = ones(pspace, pspace)
    block(n, Irrep[SU₂](0)) .= 0
    n
end

"""Two endpoints for Sᵢ⋅Sⱼ; does not include -nᵢnⱼ/4."""
const SS = let
    aspace = Rep[SU₂](1 => 1)
    SL = ones(pspace, pspace ⊗ aspace) * sqrt(3) / 2
    SR = permute(SL', ((2, 1), (3,)))
    SL, SR
end

"""
Three endpoints for -im * Sᵢ⋅(Sⱼ×Sₖ), using the same real-tensor convention
as U1SU2tJBoson. Multiply by im to obtain physical scalar chirality.
"""
const SSS = let
    aspace = Rep[SU₂](1 => 1)
    SL = ones(pspace, pspace ⊗ aspace)
    SM = zeros(aspace ⊗ pspace, pspace ⊗ aspace)
    block(SM, Irrep[SU₂](1 // 2)) .= 3 / 4
    block(SM, Irrep[SU₂](3 // 2)) .= 3 / 8
    SR = ones(aspace ⊗ pspace, pspace)
    SL, SM, SR
end

"""Two endpoints for ∑σ b†ᵢσ bⱼσ, with auxiliary spin 1/2."""
const BdagB = let
    aspace = Rep[SU₂](1 // 2 => 1)
    # SU(2) permits both particle-number directions. Select only creation on
    # the left and annihilation on the right, avoiding pair-source terms.
    Bdag = zeros(pspace, pspace ⊗ aspace)
    block(Bdag, Irrep[SU₂](1 // 2))[1, 1] = 1
    B = zeros(aspace ⊗ pspace, pspace)
    block(B, Irrep[SU₂](1 // 2))[1, 1] = 1
    Bdag, B
end

"""Two endpoints for ∑σ bᵢσ b†ⱼσ; equals BdagB(j,i) for i ≠ j."""
const BBdag = let
    aspace = Rep[SU₂](1 // 2 => 1)
    iso = isometry(aspace, flip(aspace))
    @tensor B[a; c d] := BdagB[1]'[a, b, c] * iso'[d, b]
    @tensor Bdag[d a; c] := BdagB[2]'[a, b, c] * iso[b, d]
    B, Bdag
end

"""
Two endpoints for [Δₛ(i,j)]†, where
Δₛ(i,j) = (bᵢ↓ bⱼ↑ - bᵢ↑ bⱼ↓)/√2 and Δₛ(j,i) = -Δₛ(i,j).
A singlet pair has spin 0, so no open symmetry-carrying leg is needed.
"""
const Δₛdag = let
    A = BdagB[1]
    aspace = Rep[SU₂](1 // 2 => 1)
    # Align the SU(2) duality phase with the explicit ↓↑ - ↑↓ convention.
    iso = -isometry(aspace, flip(aspace)) / sqrt(2)
    @tensor B[d a; b] := A[a b c] * iso[c d]
    A, B
end

"""Two endpoints for Δₛ(i,j), with the same ordered sites as Δₛdag."""
const Δₛ = let (A, B) = Δₛdag
    C = permute(B', ((1,), (3, 2)))
    D = -permute(A', ((2, 1), (3,)))
    C, D
end

"""Four bosonic endpoints for Δₛ†(i,j) Δₛ(k,l), in site order (i,j,k,l)."""
const ΔₛdagΔₛ = (Δₛdag..., Δₛ...)

"""
Four endpoints for ∑ₘ Δₜ,ₘ†(i,j) Δₜ,ₘ(k,l), in that site order, where
    Δₜ,+1(i,j) = bᵢ↑ bⱼ↑,
    Δₜ, 0(i,j) = (bᵢ↑ bⱼ↓ + bᵢ↓ bⱼ↑) / √2,
    Δₜ,-1(i,j) = bᵢ↓ bⱼ↓.
Bosonic triplet pairs are symmetric under exchange of the sites. The middle
auxiliary leg carries spin 1. There is no extra factor of three.
"""
const ΔₜdagΔₜ = let
    A = BdagB[1]
    aspace = Rep[SU₂](1 // 2 => 1)
    pairspace = Rep[SU₂](1 => 1)
    iso = isometry(aspace ⊗ aspace, pairspace)
    @tensor B[d a; b e] := A[a b c] * iso[c d e]
    C = permute(B', ((2, 1), (4, 3)))
    D = permute(A', ((2, 1), (3,)))
    A, B, C, D
end

end # module SU₂tJBoson

"""ASCII alias of SU₂tJBoson."""
const SU2tJBoson = SU₂tJBoson
