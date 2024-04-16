function SU₂Heisenberg(Latt::AbstractLattice; J::Number=1.0, J′::Number=0.0)

     # note J > 0 means AFM

     Root = InteractionTreeNode(IdentityOperator(0), nothing)

     let LocalSpace = SU₂Spin

          # NN interaction
          for pairs in neighbor(Latt)
               addIntr2!(Root, LocalSpace.SS, pairs,
                    J; name=(:S, :S))
          end

          # NNN interaction
          for pairs in neighbor(Latt; level = 2)
               addIntr2!(Root, LocalSpace.SS, pairs,
                    J′; name=(:S, :S))
          end

     end

     return InteractionTree(Root)

end

function U₁Heisenberg(Latt::AbstractLattice; J::Number=1, J′::Number=0)

    # note J > 0 means AFM

    Root = InteractionTreeNode(IdentityOperator(0), nothing)

    let LocalSpace = U₁Spin

         # NN interaction
         for pairs in neighbor(Latt)
              addIntr2!(Root, (LocalSpace.Sz, LocalSpace.Sz), pairs,
                   J; name=(:Sz, :Sz))
              addIntr2!(Root, LocalSpace.S₋₊, pairs,
                   J/2; name=(:Sm, :Sp))
              addIntr2!(Root, LocalSpace.S₊₋, pairs,
                   J/2; name=(:Sp, :Sm))
         end

        # NNN interaction
        for pairs in neighbor(Latt; level = 2)
            addIntr2!(Root, (LocalSpace.Sz, LocalSpace.Sz), pairs,
                    J′; name=(:Sz, :Sz))
            addIntr2!(Root, LocalSpace.S₋₊, pairs,
                   J′/2; name=(:Sm, :Sp))
            addIntr2!(Root, LocalSpace.S₊₋, pairs,
                   J′/2; name=(:Sp, :Sm))
        end

    end

    return InteractionTree(Root)

end