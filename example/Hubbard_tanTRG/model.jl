function U₁SU₂Hubbard(Latt::AbstractLattice; t::Number=1, t′::Number=0, U::Number=8, V::Number=0, μ::Number=0)

     Root = InteractionTreeNode(IdentityOperator(0), nothing)

     let LocalSpace = U₁SU₂Fermion
          # NN hopping
          for pairs in neighbor(Latt; ordered=true)
               addIntr!(Root, LocalSpace.FdagF, pairs, (true, true),
                    -t; Z=LocalSpace.Z, name=(:Fdag, :F))
          end
          # NNN hopping
          for pairs in neighbor(Latt; level = 2, ordered=true)
               addIntr!(Root, LocalSpace.FdagF, pairs, (true, true),
                    -t′; Z=LocalSpace.Z, name=(:Fdag, :F))
          end
          # nn interaction
          for pairs in neighbor(Latt)
               addIntr!(Root, (LocalSpace.n, LocalSpace.n), pairs, (false, false),
                    V; name=(:n, :n))
          end

          # U and μ
          for i in 1:size(Latt)
               addIntr!(Root, LocalSpace.nd, i, U; name=:nd)
               addIntr!(Root, LocalSpace.n, i, -μ; name=:n)
          end
     end

     return InteractionTree(Root)

end
