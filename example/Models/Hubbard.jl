function U₁SU₂Hubbard(Latt::AbstractLattice; t::Number=1, t′::Number=0, U::Number=8, V::Number=0, μ::Number=0)

     Root = InteractionTreeNode(IdentityOperator(0), nothing)

     let LocalSpace = U₁SU₂Fermion
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
          # nn interaction
          for pairs in neighbor(Latt)
               addIntr2!(Root, (LocalSpace.n, LocalSpace.n), pairs,
                    V; name=(:n, :n))
          end

          # U and μ
          for i in 1:length(Latt)
               addIntr1!(Root, LocalSpace.nd, i, U; name=:nd)
               addIntr1!(Root, LocalSpace.n, i, -μ; name=:n)
          end
     end

     return InteractionTree(Root)

end

function ℤ₂SU₂Hubbard(Latt::AbstractLattice; t::Number=1, t′::Number=0, U::Number=8, V::Number=0, μ::Number=0, hd::Number=0, hs::Number=0)

     @assert hd == 0 || hs == 0 
     if hd != 0 || hs != 0
          @assert isa(Latt, SquareLattice)
     end

     Root = InteractionTreeNode(IdentityOperator(0), nothing)
     Root_N = InteractionTreeNode(IdentityOperator(0), nothing)
     Root_h = InteractionTreeNode(IdentityOperator(0), nothing)
     let LocalSpace = ℤ₂SU₂Fermion
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
          # nn interaction
          for pairs in neighbor(Latt)
               addIntr2!(Root, (LocalSpace.n, LocalSpace.n), pairs,
                    V; name=(:n, :n))
          end

          # U and μ
          for i in 1:length(Latt)
               addIntr1!(Root, LocalSpace.nd, i, U; name=:nd)
               addIntr1!(Root_N, LocalSpace.n, i, -μ; name=:n)
          end

          # singlet pairing field
          # note singlet pairing is symmetric under exchange of two sites
          for pairs in neighbor(Latt)
               f = Latt[pairs[1]][1] == Latt[pairs[2]][1] ? -1 : 1
               addIntr2!(Root_h, LocalSpace.Δₛ, pairs, -(hd*f + hs)/2; Z=LocalSpace.Z, name=(:F, :F))
               addIntr2!(Root_h, LocalSpace.Δₛdag, pairs, -(hd*f + hs)/2; Z=LocalSpace.Z, name=(:Fdag, :Fdag))
          end

     end

     return InteractionTree(Root, Root_N, Root_h)

end