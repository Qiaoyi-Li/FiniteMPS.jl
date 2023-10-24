function U₁SU₂Hubbard(Latt::AbstractLattice; t::Number=1, t′::Number=0, U::Number=8, V::Number=0, μ::Number=0)

     Root = InteractionTreeNode(IdentityOperator(0), nothing)
     # NN hopping
     for pairs in neighbor(Latt; ordered = true)
          addIntr2!(Root, U₁SU₂Fermion.FdagF, pairs,
          -t; Z=U₁SU₂Fermion.Z, name = (:Fdag, :F))
     end
     # NNN hopping
     for pairs in neighbor(Latt; r = (1,1), ordered = true)
          addIntr2!(Root, U₁SU₂Fermion.FdagF, pairs,
               -t′; Z=U₁SU₂Fermion.Z, name = (:Fdag, :F))
     end
     # nn interaction
     for pairs in neighbor(Latt)
          addIntr2!(Root, (U₁SU₂Fermion.n, U₁SU₂Fermion.n), pairs,
               V; name = (:n, :n))
     end

     # U and μ
     for i in 1:length(Latt)
          addIntr1!(Root, U₁SU₂Fermion.nd, i, U; name = :nd)
          addIntr1!(Root, U₁SU₂Fermion.n, i, -μ; name = :n)
     end

     return InteractionTree(Root)

end