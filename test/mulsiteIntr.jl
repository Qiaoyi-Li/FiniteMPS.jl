using BenchmarkFreeFermions
using Combinatorics

L = 8 # L must be even here
D = 32
tol = 1e-8
duplicated = false

# generate a random TB model
Tij = rand(ComplexF64, L, L)
Tij += Tij'

# exact results
ϵ, V = EigenModes(Tij)

# =============== DMRG ===============
Root = InteractionTreeNode()
for i in 1:L, j in 1:L
	if i == j
		addIntr1!(Root, U1SpinlessFermion.n, i, -Tij[i, i]; name = :n)
	else
		addIntr2!(Root, U1SpinlessFermion.FdagF, (i, j), -Tij[i, j]; name = (:Fdag, :F), Z = U1SpinlessFermion.Z)
	end
end
H = AutomataMPO(InteractionTree(Root))
Ψ = randMPS(ComplexF64, L, U1SpinlessFermion.pspace, Rep[U₁](c => 1 for c in -1:1/2:1))
Env = Environment(Ψ', H, Ψ)
lsEg = zeros(20)
for i in 1:length(lsEg)
	info, _ = DMRGSweep2!(Env; trunc = truncdim(D))
	lsEg[i] = info[2][1].Eg
end
Eg = lsEg[end]
errEg = Eg - sum(ϵ[1:div(L, 2)])
@test abs(errEg) < tol # make sure DMRG works


# =============== calculate observables ===============
Tree = ObservableTree()
# 2-site 
for i in 1:L, j in 1:L
	!duplicated && i == j && continue
	addObs!(Tree, U1SpinlessFermion.FFdag, (i, j);
		Z = U1SpinlessFermion.Z, name = (:F, :Fdag))
end
# 4-site 
for i in 1:L, j in 1:L, k in 1:L, l in 1:L
	!duplicated && !allunique([i, j, k, l]) && continue
	addObs!(Tree, (U1SpinlessFermion.FdagF..., U1SpinlessFermion.FdagF...), (i, j, k, l); Z = U1SpinlessFermion.Z, name = (:Fdag, :F, :Fdag, :F))
	addObs!(Tree, Tuple(fill(U1SpinlessFermion.n, 4)), (i, j, k, l); name = (:n, :n, :n, :n))
end

calObs!(Tree, Ψ)
Obs = convert(NamedTuple, Tree)

# ================ test ==================
# exact results
μ = (ϵ[div(L, 2)] + ϵ[div(L, 2)+1]) / 2  # half filling
ξ = ϵ .- μ
G = GreenFunction(ξ, V, Inf)

@testset "Fermion-2" begin
	for i in 1:L, j in 1:L
		!duplicated && i == j && continue
		@test haskey(Obs.FFdag, (i, j)) && abs(Obs.FFdag[(i, j)] - G[i, j]) < tol
	end
end

@testset "Fermion-4" begin
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		!duplicated && !allunique([i, j, k, l]) && continue
		O_ex = ExpectationValue(G, [i, j, k, l], [1, 3])
		@test haskey(Obs.FdagFFdagF, (i, j, k, l)) && abs(Obs.FdagFFdagF[(i, j, k, l)] - O_ex) < tol
	end
end

lsperms = permutations(1:4) |> collect 
@testset "Boson-4" begin
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		!duplicated && !allunique([i, j, k, l]) && continue
		O_ex = ExpectationValue(G, [i, i, j, j, k, k, l, l], [1, 3, 5, 7])
		# note the indices are only valid up to a permutation when the operators are the same
		sites = [i, j, k, l]
		idx = findfirst(lsperms) do perm
			permuted_sites = sites[perm]
			haskey(Obs.nnnn, Tuple(permuted_sites))
		end
		@test !isnothing(idx) && abs(Obs.nnnn[Tuple(sites[lsperms[idx]])] - O_ex) < tol
	end
end
