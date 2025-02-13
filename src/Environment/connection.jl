"""
	 connection!(Env1::SparseEnvironment, Env2::SparseEnvironment, ...; 
		  kwargs...) -> ::Matrix 

Return the connection `⟨∇⟨Hᵢ⟩, ∇⟨Hⱼ⟩⟩` where `H₀`, `H₁`, ⋯, `Hₙ` are the Hamiltonian corresponding to the given environments.

Note this function will move all the canonical centers to the right boundary. 

# kwargs
	 moveback::Bool = false
Move the canonical center back to the left boundary if `true`.
"""
function connection!(lsEnv::SparseEnvironment{L, 3, T}...;
	moveback::Bool = false) where {L, T <: Tuple{AdjointMPS, SparseMPO, DenseMPS}}
	N = length(lsEnv)
	NT = mapreduce(x -> typeof(coef(x[3])), promote_type, lsEnv)

	conMat = zeros(NT, N, N)
	lsHx = Vector{MPSTensor}(undef, N)
	lsx1 = Vector{MPSTensor}(undef, N)
	lsx0 = Vector{MPSTensor}(undef, N)
	for si in 1:L
		for n in 1:N
			canonicalize!(lsEnv[n][3], si)
			canonicalize!(lsEnv[n], si)
			PH = CompositeProjectiveHamiltonian(lsEnv[n].El[si], lsEnv[n].Er[si], (lsEnv[n][2][si],))
			if si == 1
				lsx1[n] = lsEnv[n][3][si]
			end
			lsHx[n] = action(lsx1[n], PH)
		end

		for i in 1:N, j in i:N
			conMat[i, j] += (inner(lsHx[i], lsHx[j])
							 -
							 inner(lsHx[i], lsx1[i]) * inner(lsx1[i], lsHx[j])
							 -
							 inner(lsHx[i], lsx1[j]) * inner(lsx1[j], lsHx[j])
							 +
							 inner(lsHx[i], lsx1[i]) * inner(lsx1[i], lsx1[j]) * inner(lsx1[j], lsHx[j]))
		end

		# subtract the double counting due to gauge redundancy
		if si < L

			for n in 1:N
				lsEnv[n][3][si], lsx0[n] = leftorth(lsx1[n])
				canonicalize!(lsEnv[n], si + 1, si)
				PH = CompositeProjectiveHamiltonian(lsEnv[n].El[si+1], lsEnv[n].Er[si], ())
				lsHx[n] = action(lsx0[n], PH)

                    # update next site 
                    lsx1[n] = lsx0[n] * lsEnv[n][3][si+1]
			end

			for n in 1:N
				# warning: this for loop cannot be merged with the above one as the MPS of these environments may be the same object
                    lsEnv[n][3][si+1] = lsx1[n]
				Center(lsEnv[n][3])[:] = [si + 1, si + 1]
			end

			for i in 1:N, j in i:N
				conMat[i, j] -= (inner(lsHx[i], lsHx[j])
								 -
								 inner(lsHx[i], lsx0[i]) * inner(lsx0[i], lsHx[j])
								 -
								 inner(lsHx[i], lsx0[j]) * inner(lsx0[j], lsHx[j])
								 +
								 inner(lsHx[i], lsx0[i]) * inner(lsx0[i], lsx0[j]) * inner(lsx0[j], lsHx[j])
				)
			end
		end

	end

	# fill i > j
	for i in 1:N, j in 1:i-1
		conMat[i, j] = conMat[j, i]'
	end

	if moveback
		for n in 1:N
			canonicalize!(lsEnv[n][3], 1)
			canonicalize!(lsEnv[n], 1)
		end
	end

	return conMat
end
