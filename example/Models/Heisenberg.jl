function SU₂Heisenberg(Latt::AbstractLattice; J::Number=1, J′::Number=0)

     # note J > 0 means AFM

     Root = InteractionTreeNode(IdentityOperator(0), nothing)

     # get the distance of NN and NNN pairs
     lsd = map(1:length(Latt)) do i
                metric(Latt, Latt[i])
           end |> sort |> unique
     i = 1
     while i < length(lsd)
          if lsd[i] ≈ lsd[i+1]
               deleteat!(lsd, i + 1)
          else
               i += 1
          end
     end

     let LocalSpace = SU₂Spin

          # NN interaction
          for pairs in neighbor(Latt; d=lsd[1])
               addIntr2!(Root, LocalSpace.SS, pairs,
                    J; name=(:S, :S))
          end

          # NNN interaction
          for pairs in neighbor(Latt; d=lsd[2])
               addIntr2!(Root, LocalSpace.SS, pairs,
                    J′; name=(:S, :S))
          end

     end

     return InteractionTree(Root)

end

function U₁Heisenberg(Latt::AbstractLattice; J::Number=1, J′::Number=0)

    # note J > 0 means AFM

    Root = InteractionTreeNode(IdentityOperator(0), nothing)

    # get the distance of NN and NNN pairs
    lsd = map(1:length(Latt)) do i
               metric(Latt, Latt[i])
          end |> sort |> unique
    i = 1
    while i < length(lsd)
         if lsd[i] ≈ lsd[i+1]
              deleteat!(lsd, i + 1)
         else
              i += 1
         end
    end

    let LocalSpace = U₁Spin

         # NN interaction
         for pairs in neighbor(Latt; d=lsd[1])
              addIntr2!(Root, (LocalSpace.Sz, LocalSpace.Sz), pairs,
                   J; name=(:Sz, :Sz))
              addIntr2!(Root, LocalSpace.S₋₊, pairs,
                   J/2; name=(:Sm, :Sp))
              addIntr2!(Root, LocalSpace.S₊₋, pairs,
                   J/2; name=(:Sp, :Sm))
         end

        # NNN interaction
        for pairs in neighbor(Latt; d=lsd[2])
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
