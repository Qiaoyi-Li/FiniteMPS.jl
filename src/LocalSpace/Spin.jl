"""
     module SU₂Spin

Prepare the local space of SU₂ spin-1/2.

# Fields
     pspace::VectorSpace
Local `d = 2` Hilbert space.

     SS::NTuple{2, TensorMap}
Two rank-`3` operators of Heisenberg `S⋅S` interaction.
"""
module SU₂Spin

using TensorKit

const pspace = Rep[SU₂](1//2 => 1)
# S⋅S interaction
const SS = let
     aspace = Rep[SU₂](1 => 1)
     SL = TensorMap(ones, Float64, pspace, pspace ⊗ aspace) * sqrt(3) / 2

     SR = permute(SL', ((2, 1), (3,)))
     SL, SR
end

end

"""
     const SU2Spin = SU₂Spin
"""
const SU2Spin = SU₂Spin

module U₁Spin

using TensorKit

const pspace = Rep[U₁](-1 // 2 => 1, 1 // 2 => 1)

const Sz = let
     Sz = TensorMap(ones, pspace, pspace)
     block(Sz, Irrep[U₁](1 // 2)) .= 1/2
     block(Sz, Irrep[U₁](-1 // 2)) .= -1/2
     Sz
end

# S+ S- interaction
# convention: S⋅S = SzSz + (S₊₋ + S₋₊)/2
const S₊₋ = let
     aspace = Rep[U₁](1 => 1)
     S₊ = TensorMap(ones, pspace, pspace ⊗ aspace)
     S₋ = TensorMap(ones, aspace ⊗ pspace, pspace)
     S₊, S₋
end

const S₋₊ = let
     aspace = Rep[U₁](1 => 1)
     iso = isometry(aspace, flip(aspace))
     @tensor S₋[a; c d] := S₊₋[1]'[a, b, c] * iso'[d, b]
     @tensor S₊[d a; c] := S₊₋[2]'[a, b, c] * iso[b, d]
     S₋, S₊
end

end

const U1Spin = U₁Spin
