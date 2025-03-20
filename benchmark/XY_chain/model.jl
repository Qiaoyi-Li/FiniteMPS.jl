function XYChain(L::Int64; J::Real=1.0)

	Root = InteractionTreeNode(IdentityOperator(0), nothing)

     # J(SxSx + SySy) = J/2 (S+S- + S-S+)
	for i in 1:L-1
          addIntr!(Root, U₁Spin.S₊₋, (i, i+1), (false, false), J/2; name = (:S₊, :S₋))
          addIntr!(Root, U₁Spin.S₋₊, (i, i+1), (false, false), J/2; name = (:S₋, :S₊))
	end

     return InteractionTree(Root)
end

function ExactSolution(L::Int64, lsβ::AbstractVector{<:Real}; J::Real=1.0)
     # JW transformation to spinless fermions
     T = diagm(1 => J/2*ones(L-1), -1 => J/2*ones(L-1)) 

     ϵ = SingleParticleSpectrum(T)

     return  map(β -> FreeEnergy(ϵ, β, 0.0), lsβ)
end
