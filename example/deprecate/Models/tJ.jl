function U₁SU₂tJ(Latt::AbstractLattice; t::Number=1, t′::Number=0, J::Number=1//3, J′::Number=J*abs2(t′/t), V::Number=0, μ::Number=0)

     Root = InteractionTreeNode(IdentityOperator(0), nothing)

     let LocalSpace = U₁SU₂tJFermion
          # NN hopping
          for pairs in neighbor(Latt; ordered=true)
               addIntr2!(Root, LocalSpace.FdagF, pairs,
                    -t; Z=LocalSpace.Z, name=(:Fdag, :F))
          end
          # NNN hopping
          for pairs in neighbor(Latt; r=(1, 1), ordered=true)
               addIntr2!(Root, LocalSpace.FdagF, pairs,
                    -t′; Z=LocalSpace.Z, name=(:Fdag, :F))
          end

          # J term
          for pairs in neighbor(Latt)
               addIntr2!(Root, LocalSpace.SS, pairs,
                    J; name=(:S, :S))
               addIntr2!(Root, (LocalSpace.n, LocalSpace.n), pairs,
                    -J / 4; name=(:n, :n))
          end
          for pairs in neighbor(Latt; r=(1, 1))
               addIntr2!(Root, LocalSpace.SS, pairs,
                    J′; name=(:S, :S))
               addIntr2!(Root, (LocalSpace.n, LocalSpace.n), pairs,
                    -J′ / 4; name=(:n, :n))
          end


          # additional nn interaction
          for pairs in neighbor(Latt)
               addIntr2!(Root, (LocalSpace.n, LocalSpace.n), pairs,
                    V; name=(:n, :n))
          end

          # μ
          for i in 1:length(Latt)
               addIntr1!(Root, LocalSpace.n, i, -μ; name=:n)
          end
     end


     return InteractionTree(Root)

end