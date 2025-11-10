using BenchmarkFreeFermions

# test DMRG and 4-site observables for free spinful fermions

L = 6 # L must be even here
D = 64
tol = 1e-8
duplicated = true

# generate a random TB model with SU2 spin symmetry
Tij = rand(ComplexF64, L, L)
Tij += Tij'
Tij = hcat(vcat(Tij, zeros(ComplexF64, L, L)), vcat(zeros(ComplexF64, L, L), Tij))

# exact results
ϵ, V = EigenModes(Tij)

# =============== DMRG ===============
Tree = InteractionTree(L)
for i in 1:L, j in 1:L
	if i == j
		addIntr!(Tree, U1SU2Fermion.n, i, -Tij[i, i]; name = :n)
	else
		addIntr!(Tree, U1SU2Fermion.FdagF, (i, j), (true, true),
			-Tij[i, j]; name = (:Fdag, :F), Z = U1SU2Fermion.Z)
	end
end
H = AutomataMPO(Tree)
Ψ = randMPS(ComplexF64, L, U1SU2Fermion.pspace, Rep[U₁×SU₂]((c, s) => 1 for c in -1:1:1 for s in 0:1/2:1))
Env = Environment(Ψ', H, Ψ)
lsEg = zeros(20)
for i in 1:length(lsEg)
	info, _ = DMRGSweep2!(Env; trunc = truncdim(D))
	lsEg[i] = info[2][1].Eg
end
Eg = lsEg[end]
errEg = Eg - sum(ϵ[1:L])
@test abs(errEg) < tol # make sure DMRG works


# =============== calculate observables ===============
Tree = ObservableTree(L)
# 4-site 
for i in 1:L, j in 1:L, k in 1:L, l in 1:L
	!duplicated && !allunique([i, j, k, l]) && continue
	# singlet pairing 
	addObs!(Tree, U1SU2Fermion.ΔₛdagΔₛ, (i, j, k, l), (true, true, true, true); Z = U1SU2Fermion.Z, name = (:Fdag, :FdagS, :FS, :F), IntrName = :SSC)
	# triplet pairing
	addObs!(Tree, U1SU2Fermion.ΔₜdagΔₜ, (i, j, k, l), (true, true, true, true); Z = U1SU2Fermion.Z, name = (:Fdag, :FdagT, :FT, :F), IntrName = :TSC)

	# charge bond operator
	addObs!(Tree, (U1SU2Fermion.FdagF..., U1SU2Fermion.FdagF...), (i, j, k, l), (true, true, true, true); Z = U1SU2Fermion.Z, name = (:Fdag, :F, :Fdag, :F), IntrName = :CB)
	# spin bond operator
	addObs!(Tree, U1SU2Fermion.SBSB, (i, j, k, l), (true, true, true, true); Z = U1SU2Fermion.Z, name = (:Fdag, :FS, :FdagS, :F), IntrName = :SB)
end

calObs!(Tree, Ψ)
Obs = convert(NamedTuple, Tree)

# ================ test ==================
# exact results
μ = (ϵ[L] + ϵ[L+1]) / 2  # half filling
ξ = ϵ .- μ
G = GreenFunction(ξ, V, Inf)

# singlet pairing
@testset "SSC" begin
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		!duplicated && !allunique([i, j, k, l]) && continue
		# (c_i↑^dag c_j↓^dag - c_i↓^dag c_j↑^dag)(c_k↓ c_l↑ - c_k↑ c_l↓) / 2 
		# = c_i↑^dag c_j↓^dag c_k↓ c_l↑ - c_i↑^dag c_j↓^dag c_k↑ c_l↓

		O_ex = ExpectationValue(G, [i, j+L, k+L, l], [1, 2]) - ExpectationValue(G, [i, j+L, k, l+L], [1, 2])
		@test haskey(Obs.SSC, (i, j, k, l)) && abs(Obs.SSC[(i, j, k, l)] - O_ex) < tol
	end
end

# triplet pairing
@testset "TSC" begin
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		!duplicated && !allunique([i, j, k, l]) && continue
		# 3 c_i↑^dag c_j↑^dag c_l↑ c_k↑
		O_ex = 3 * ExpectationValue(G, [i, j, l, k], [1, 2])
		@test haskey(Obs.TSC, (i, j, k, l)) && abs(Obs.TSC[(i, j, k, l)] - O_ex) < tol
	end
end

# charge bond operator
@testset "CB" begin
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		!duplicated && !allunique([i, j, k, l]) && continue 
		# (c_i↑^dag c_j↑ + c_i↓^dag c_j↓)(c_k↑^dag c_l↑ + c_k↓^dag c_l↓)
		# = 2(c_i↑^dag c_j↑ c_k↑^dag c_l↑ + c_i↑^dag c_j↑ c_k↓^dag c_l↓)
		O_ex = 2 * (ExpectationValue(G, [i, j, k, l], [1, 3]) + ExpectationValue(G, [i, j, k+L, l+L], [1, 3]))
		@test haskey(Obs.CB, (i, j, k, l)) && abs(Obs.CB[(i, j, k, l)] - O_ex) < tol
	end
end

# spin bond operator
@testset "SB" begin
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		!duplicated && !allunique([i, j, k, l]) && continue 
		# 3 * (c_i↑^dag c_j↓)(c_k↓^dag c_l↑) / 2
		O_ex = 3/2 * ExpectationValue(G, [i, j+L, k+L, l], [1, 3])
		@test haskey(Obs.SB, (i, j, k, l)) && abs(Obs.SB[(i, j, k, l)] - O_ex) < tol
	end
end





