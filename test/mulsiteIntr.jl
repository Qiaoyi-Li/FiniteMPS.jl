using BenchmarkFreeFermions

L = 8 # L must be even here
D = 32
tol = 1e-8
duplicated = true

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
# 4-site 
for i in 1:L, j in 1:L, k in 1:L, l in 1:L
	!duplicated && !allunique([i, j, k, l]) && continue
	addObs!(Tree, (U1SpinlessFermion.FdagF..., U1SpinlessFermion.FdagF...), (i, j, k, l), (true, true, true, true); Z = U1SpinlessFermion.Z, name = (:Fdag, :F, :Fdag, :F))
	addObs!(Tree, Tuple(fill(U1SpinlessFermion.n, 4)), (i, j, k, l), (false, false, false, false); name = (:n, :n, :n, :n))
end
# 3-site 
for i in 1:L, j in 1:L, k in 1:L
	!duplicated && !allunique([i, j, k]) && continue
	addObs!(Tree, (U1SpinlessFermion.n, U1SpinlessFermion.n, U1SpinlessFermion.n), (i, j, k), (false, false, false); name = (:n, :n, :n))
	addObs!(Tree, (U1SpinlessFermion.FdagF..., U1SpinlessFermion.n), (i, j, k), (true, true, false); Z = U1SpinlessFermion.Z, name = (:Fdag, :F, :n))
end
# 2-site 
for i in 1:L, j in 1:L
	!duplicated && i == j && continue
	addObs!(Tree, U1SpinlessFermion.FFdag, (i, j), (true, true);
		Z = U1SpinlessFermion.Z, name = (:F, :Fdag))
	addObs!(Tree, U1SpinlessFermion.FdagF, (i, j), (true, true);
		Z = U1SpinlessFermion.Z, name = (:Fdag, :F))
	addObs!(Tree, (U1SpinlessFermion.n, U1SpinlessFermion.n), (i, j), (false, false); name = (:n, :n))
end
# 1-site
for i in 1:L
	addObs!(Tree, U1SpinlessFermion.n, i; name = :n)
end

calObs!(Tree, Ψ)
Obs = convert(NamedTuple, Tree)

# ================ test ==================
# exact results
μ = (ϵ[div(L, 2)] + ϵ[div(L, 2)+1]) / 2  # half filling
ξ = ϵ .- μ
G = GreenFunction(ξ, V, Inf)

@testset "Boson-1" begin
	for i in 1:L
		@test haskey(Obs.n, (i,)) && abs(Obs.n[(i,)] - (1 - G[i, i])) < tol
	end
end

@testset "Boson-2" begin
	for i in 1:L, j in 1:L
		!duplicated && i == j && continue
		O_ex = ExpectationValue(G, [i, i, j, j], [1, 3])
		@test haskey(Obs.nn, (i, j)) && abs(Obs.nn[(i, j)] - O_ex) < tol
	end
end

@testset "Fermion-2" begin
	for i in 1:L, j in 1:L
		!duplicated && i == j && continue
		@test haskey(Obs.FFdag, (i, j)) && abs(Obs.FFdag[(i, j)] - G[i, j]) < tol
		O_ex = i == j ? 1 - G[i, i] : -G[j, i]
		@test haskey(Obs.FdagF, (i, j)) && abs(Obs.FdagF[(i, j)] - O_ex) < tol
	end
end

@testset "Boson-3" begin
	for i in 1:L, j in 1:L, k in 1:L
		!duplicated && !allunique([i, j, k]) && continue
		O_ex = ExpectationValue(G, [i, i, j, j, k, k], [1, 3, 5])
		@test haskey(Obs.nnn, (i, j, k)) && abs(Obs.nnn[(i, j, k)] - O_ex) < tol
	end
end

@testset "Boson-1-Fermion-2" begin
	for i in 1:L, j in 1:L, k in 1:L
		!duplicated && !allunique([i, j, k]) && continue
		O_ex = ExpectationValue(G, [i, j, k, k], [1, 3])
		@test haskey(Obs.FdagFn, (i, j, k)) && abs(Obs.FdagFn[(i, j, k)] - O_ex) < tol
	end
end

@testset "Boson-4" begin
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		!duplicated && !allunique([i, j, k, l]) && continue
		O_ex = ExpectationValue(G, [i, i, j, j, k, k, l, l], [1, 3, 5, 7])
		@test haskey(Obs.nnnn, (i, j, k, l)) && abs(Obs.nnnn[(i, j, k, l)] - O_ex) < tol
	end
end

@testset "Fermion-4" begin
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		!duplicated && !allunique([i, j, k, l]) && continue
		O_ex = ExpectationValue(G, [i, j, k, l], [1, 3])
		@test haskey(Obs.FdagFFdagF, (i, j, k, l)) && abs(Obs.FdagFFdagF[(i, j, k, l)] - O_ex) < tol
	end
end



