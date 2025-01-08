function SpinlessFreeFermion(Latt::AbstractLattice; t::Number=1, t′::Number=0, μ::Number=0)

     L = size(Latt)
     T = zeros(L, L)
     for pairs in neighbor(Latt; ordered = true)
          T[pairs...] = t
     end
     for pairs in neighbor(Latt; level = 2, ordered = true)
          T[pairs...] = t′
     end
     for i in 1:L
          T[i, i] = μ
     end
     return SpinlessFreeFermion(T)
end

function SpinlessFreeFermion(T::AbstractMatrix)
     # - sum_{i,j} T_ij c_i^† c_j 

     L = size(T, 1)
     @assert size(T, 2) == L

     Root = InteractionTreeNode(IdentityOperator(0), nothing)

     # hopping
     for i = 1:L, j = i+1:L
          @assert T[i, j] == T[j, i]'
          addIntr!(Root, U₁SpinlessFermion.FdagF, (i, j), (true, true),
               -T[i, j]; Z = U₁SpinlessFermion.Z, name = (:Fdag, :F))
          addIntr!(Root, U₁SpinlessFermion.FdagF, (j, i), (true, true),
               -T[j, i]; Z = U₁SpinlessFermion.Z, name = (:Fdag, :F))
     end

     # chemical potential
     for i = 1:L
          addIntr!(Root, U₁SpinlessFermion.n, i, -T[i, i]; name = :n)
     end

     return InteractionTree(Root)
end

function ExactSolution(Latt::AbstractLattice, lsβ::Vector{Float64}; t::Number=1, t′::Number=0, μ::Number=0)
     L = size(Latt)
     T = zeros(L, L)
     for pairs in neighbor(Latt; ordered = true)
          T[pairs...] = t
     end
     for pairs in neighbor(Latt; level = 2, ordered = true)
          T[pairs...] = t′
     end
     for i in 1:L
          T[i, i] = μ
     end

     ϵ, _ = eigen(-T)
     lsF = similar(lsβ)
     lsE = similar(lsβ)
     for (i, β) in enumerate(lsβ)
          lsF[i] = - sum(x -> log(1 + exp(-β*x)), ϵ) / β
          lsE[i] = sum(x -> x / (1 + exp(β*x)), ϵ)
     end

     return lsF, lsE
end