"""
     AutomataMPO(Tree::InteractionTree, L::Int64) -> ::SparseMPO

Convert an interaction tree to a sparse MPO.
"""
function AutomataMPO(Tree::InteractionTree, L::Int64 = treeheight(Tree.Root) - 1)
     # convert an interaction tree to a sparse MPO
     Root = Tree.Root

     # count_size
     D = zeros(Int64, L + 1) # D[i] denotes the bond dimension from i-1 to i
     for node in PreOrderDFS(Root)
          isnothing(node.Op) && continue
          isempty(node.children) && continue
          D[node.Op.si+1] += 1
     end

     # additional channel to store the accumulation
     D[2:end] .+= 1

     H = Vector{SparseMPOTensor}(undef, L)
     for i = 1:L
          H[i] = SparseMPOTensor(nothing, D[i], D[i+1])
     end

     c = vcat(0, repeat([1,], L))
     for node in PreOrderDFS(Root)
          isnothing(node.Op) && continue
          si = node.Op.si
          if si == 0
               c[1] += 1
               continue
          end

          # merge this channel to accumulation
          if !isnan(node.Op.strength)
               H[si][c[si], 1] += deepcopy(node.Op)
          end

          # propagate
          if !isempty(node.children)
               c[si+1] += 1
               H[si][c[si], c[si+1]] = _convertStrength(node.Op)
          end

     end

     # remember the identity to propagate accumulation
     for si = 2:L
          idx = findfirst(!isnothing, H[si])
          pspace = isnothing(idx) ? nothing : getPhysSpace(H[si][idx])
          H[si][1,1] = IdentityOperator(pspace, si, 1)
     end

     return SparseMPO(H)
end

function _convertStrength(A::AbstractLocalOperator)
     B = deepcopy(A)
     B.strength = 1
     return B
end