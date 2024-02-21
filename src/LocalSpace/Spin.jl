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