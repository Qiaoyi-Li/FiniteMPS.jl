function SU₂Heisenberg(Latt::AbstractLattice; J::Number=1.0, J′::Number=0.0)

     # note J > 0 means AFM

     Tree = InteractionTree(size(Latt))

     let LocalSpace = SU₂Spin

          # NN interaction
          for pairs in neighbor(Latt)
               addIntr!(Tree, LocalSpace.SS, pairs, (false, false),
                    J; name=(:S, :S))
          end

          # NNN interaction
          for pairs in neighbor(Latt; level = 2)
               addIntr!(Tree, LocalSpace.SS, pairs, (false, false),
                    J′; name=(:S, :S))
          end

     end

     return merge!(Tree)

end

function U₁Heisenberg(Latt::AbstractLattice; J::Number=1, J′::Number=0)

    # note J > 0 means AFM

    Tree = InteractionTree(size(Latt))

    let LocalSpace = U₁Spin

         # NN interaction
         for pairs in neighbor(Latt)
              addIntr!(Tree, (LocalSpace.Sz, LocalSpace.Sz), pairs, (false, false),
                   J; name=(:Sz, :Sz))
              addIntr!(Tree, LocalSpace.S₋₊, pairs, (false, false),
                   J/2; name=(:Sm, :Sp))
              addIntr!(Tree, LocalSpace.S₊₋, pairs, (false, false),
                   J/2; name=(:Sp, :Sm))
         end

        # NNN interaction
        for pairs in neighbor(Latt; level = 2)
            addIntr!(Tree, (LocalSpace.Sz, LocalSpace.Sz), pairs, (false, false),
                    J′; name=(:Sz, :Sz))
            addIntr!(Tree, LocalSpace.S₋₊, pairs, (false, false),
                   J′/2; name=(:Sm, :Sp))
            addIntr!(Tree, LocalSpace.S₊₋, pairs, (false, false),
                   J′/2; name=(:Sp, :Sm))
        end

    end

    return merge!(Tree)

end