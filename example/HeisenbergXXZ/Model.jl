# Hamiltonian of Heisenberg XXZ model 
# H = sum_{ij} J_{ij} (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j) - h_z \sum_i S^z_i

"""
     HeisenbergXXZ(Latt::AbstractLattice;
          LocalSpace::Module = U₁Spin,
	     J::Real = 1.0,
	     J′::Real = 0.0,
	     Δ::Real = 1.0,
	     hz::Real = 0.0)
     ) -> ::InteractionTree

Return the interaction tree for the Heisenberg XXZ model. `LocalSpace ∈ [NoSymSpinOneHalf, U₁Spin, SU₂Spin]` indicates the symmetry used.
"""
function HeisenbergXXZ(Latt::AbstractLattice;
	LocalSpace::Module = U₁Spin,
	J::Real = 1.0,
	J′::Real = 0.0,
	Δ::Real = 1.0,
	hz::Real = 0.0)

	# supported LocalSpace:
	@assert in(LocalSpace, [NoSymSpinOneHalf, U₁Spin, SU₂Spin]) "supported LocalSpace: NoSymSpinOneHalf, U₁Spin, SU₂Spin!"

	Tree = InteractionTree(size(Latt))

	# Heisenberg XXZ interactions up to NNN
	for (level, J_level) in zip([1, 2], [J, J′])
		for (i, j) in neighbor(Latt; level = level)
			if LocalSpace == SU₂Spin
				@assert Δ == 1
				addIntr!(Tree, LocalSpace.SS, (i, j), (false, false), J_level; name = (:S, :S))
			elseif LocalSpace == U₁Spin
				addIntr!(Tree, LocalSpace.S₊₋, (i, j), (false, false), J_level / 2; name = (:S₊, :S₋))
				addIntr!(Tree, LocalSpace.S₋₊, (i, j), (false, false), J_level / 2; name = (:S₋, :S₊))
				addIntr!(Tree, (LocalSpace.Sz, LocalSpace.Sz), (i, j), (false, false), J_level * Δ; name = (:Sz, :Sz))

			elseif LocalSpace == NoSymSpinOneHalf
				# use S+S- to avoid ComplexF64 in Sy 
				addIntr!(Tree, (LocalSpace.S₊, LocalSpace.S₋), (i, j), (false, false), J_level / 2; name = (:S₊, :S₋))
				addIntr!(Tree, (LocalSpace.S₋, LocalSpace.S₊), (i, j), (false, false), J_level / 2; name = (:S₋, :S₊))
				addIntr!(Tree, (LocalSpace.Sz, LocalSpace.Sz), (i, j), (false, false), J_level * Δ; name = (:Sz, :Sz))
			else
				@assert false
			end
		end
	end

	# hz 
	if !iszero(hz)
          @assert LocalSpace ≠ SU₂Spin "hz must be zero in SU2 case!"
		for i in size(Latt)
               addIntr!(Tree, LocalSpace.Sz, i, -hz; name = :Sz)
		end
	end

     return merge!(Tree)
end

