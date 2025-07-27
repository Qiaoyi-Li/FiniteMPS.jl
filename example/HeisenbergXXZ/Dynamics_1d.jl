# compute the transverse DSF S^{+-}(k, ω) of the Heisenberg chain 

using FiniteMPS, FiniteLattices, CairoMakie, Statistics, NumericalIntegration

include("Model.jl")

LocalSpace = U1Spin # NoSymSpinOneHalf, U1Spin, SU2Spin
L = 64
J′ = 0.0
Δ = 1.0 # XXZ anisotropy: Sxx + Syy + ΔSzz
D = 256
etol = 1e-8 # tolerance of per site energy
LocalSpace == SU2Spin && @assert Δ == 1.0
j_ref = div(L, 2) # reference site for S_j^-

lst = 0:0.5:10 # time list
lsω = 0:0.1:4 # frequency list

# generate a 1d chain lattice, as a special case of square lattice
Latt = OpenSqua(L, 1)

# Hamiltonian MPO 
H = AutomataMPO(HeisenbergXXZ(Latt; LocalSpace = LocalSpace, J′ = J′, Δ = Δ))

# initial a random state
if LocalSpace == SU2Spin
	aspace = Rep[SU₂](i => 1 for i in 0:1/2:1)
elseif LocalSpace == U1Spin
	aspace = Rep[U₁](i => 1 for i in -1:1/2:1)
else
	aspace = ℂ^2
end
Ψ = randMPS(size(Latt), LocalSpace.pspace, aspace)

# generate the trilayer environment
Env = Environment(Ψ', H, Ψ)
lsE = [scalar!(Env)] # initial energy

# CBE+noise 
noise = 0.1
for nsweep in 1:20
	info, _ = DMRGSweep1!(Env;
		noise = (0.5, noise), # (expand ratio, noise strength)
		CBEAlg = NaiveCBE(D + div(D, 2), 1e-8; rsvd = true),
		verbose = 1, GCsweep = true,
		trunc = truncdim(D) & truncbelow(1e-12),
		K = 16)
	push!(lsE, info[2].dmrg[1].Eg)

	@show nsweep, noise, lsE[end]

	if !iszero(noise)
		if noise > 1e-4
			noise /= 10 # decrease noise
		else
			noise = 0
		end
	elseif lsE[end-1] - lsE[end] < etol * size(Latt)
		break
	end
end
@show Eg = lsE[end]

# compute S_{i,j}(t) = ⟨Ψ|e^{iHt}S_i^+ e^{-iHt}S_j^-|Ψ⟩
# = e^{iE_gt}⟨Ψ|S_i^+|Φ(t)⟩ where |Φ(t)⟩ = e^{-iHt}S_j^-|Ψ⟩

# fuse the auxiliary bonds of MPS and operators
if LocalSpace == SU2Spin
	Op_SS = LocalSpace.SS
	aspace = Rep[SU₂](i => 1 for i in 0:1/2:1) # bulk space of the initial random MPS
     aspace_S = codomain(Op_SS[2])[1] # auxiliary space of the S_i^- operator
elseif LocalSpace == U1Spin
	Op_SS = LocalSpace.S₊₋
	aspace = Rep[U₁](i => 1 for i in -1:1/2:1)
     aspace_S = codomain(Op_SS[2])[1]
else
	Op_SS = (LocalSpace.S₊, LocalSpace.S₋)
	aspace = ℂ^2
     aspace_S = ℂ^1
end
aspace_Ψ = codomain(Ψ[1])[1] # auxiliary space of the initial random MPS
aspace_Φ = fuse(aspace_S, aspace_Ψ) # auxiliary space of the target MPS |Φ⟩

# obtain S_j^-|Ψ⟩ 
Φ = let
	# wrap the operator to a MPO 
	Tree = InteractionTree(size(Latt))
	addIntr!(Tree, Op_SS[2], j_ref, 1.0; name = :S₋)
	S_MPO = AutomataMPO(Tree)

	# note Φ should be a complex MPS
	Φ = randMPS(ComplexF64, fill(LocalSpace.pspace, size(Latt)), vcat(aspace_Φ, fill(aspace, size(Latt) - 1)))

	# obtain S_MPO * Φ variationally 
	mul!(Φ, S_MPO, Ψ;
		CBEAlg = NaiveCBE(D + div(D, 4), 1e-8; rsvd = true),
		trunc = truncdim(D) & truncbelow(1e-12), verbose = 1,
		GCsweep = true, lsnoise = [0.1, 0.01, 0.001],
          tol = 1e-12,
	)
end

matSijt = zeros(ComplexF64, L, length(lst)) # S_{i,j_ref}(t) 
# ObsTree for computing correlations
Obs = ObservableTree(size(Latt))
for i in 1:size(Latt)
	addObs!(Obs, Op_SS[1], i; name = :S₊)
end
if LocalSpace == NoSymSpinOneHalf
	El = id(ℂ^1)
else
	iso = isometry(aspace_Φ, aspace_S ⊗ aspace_Ψ)
	El = permute(iso', ((2, 1), (3,)))
end
calObs!(Obs, Ψ, Φ; El = El)
matSijt[:, 1] = [Obs.Refs["S₊"][(i,)][] for i in 1:size(Latt)]

# consistency check, ⟨Ψ|S_i^+|Φ⟩ = ⟨Ψ|S_i^+ S_j^-|Ψ⟩
lserr = let
	Obs2 = ObservableTree(size(Latt))
	for i in 1:size(Latt)
		addObs!(Obs2, Op_SS, (i, j_ref), (false, false); name = (:S₊, :S₋))
	end
	calObs!(Obs2, Ψ)

	map(1:size(Latt)) do i
		abs(Obs2.Refs["S₊S₋"][(i, j_ref)][] - Obs.Refs["S₊"][(i,)][])
	end
end
@show maximum(lserr)

# construct the trilayer environment for TDVP 
Env = Environment(Φ', H, Φ)

# time evolution
for idx in 2:length(lst)
	dt = lst[idx] - lst[idx-1]

	TDVPSweep1!(Env, -im * dt;
		CBEAlg = NaiveCBE(D + div(D, 4), 1e-8; rsvd = true),
		GCsweep = true, verbose = 1,
		trunc = truncdim(D) & truncbelow(1e-12),
	)

	calObs!(Obs, Ψ, Φ; El = El)
	# e^{iEgt}⟨Ψ|S_i^+|Φ⟩
	matSijt[:, idx] = exp(im * Eg * lst[idx]) * map(1:size(Latt)) do i
		Obs.Refs["S₊"][(i,)][]
	end
	@show lst[idx]

end

# spatial FT, S(k,t) ≈ \sum_i S_{i,j_ref}(t) e^{-ik(i - j_ref)} 
lsk = 0:2/L:2 # unit = π
matcoef = map([(k, i) for k in lsk, i in 1:size(Latt)]) do (k, i)
	exp(-im * k * (i - j_ref) * π)
end
matSkt = matcoef * matSijt

# S(k, ω) = \int_{-\infty}^{\infty} S(k, t) e^{iωt} W(t) dt 
# = 2 \int_0^{\infty} Re[S(k, t)*exp(iωt)] W(t) dt
function ParzenWindow(x::Float64)
	if abs(x) < 1 / 2
		return 1 - 6 * x^2 + 6 * abs(x)^3
	elseif abs(x) < 1
		return 2 * (1 - abs(x))^3
	else
		return 0.0
	end
end
lsWt = ParzenWindow.(lst ./ maximum(lst))

matSkω = zeros(length(lsk), length(lsω))
for iω in 1:length(lsω)
	lscoef = exp.(im * lsω[iω] .* lst) .* lsWt
	for ik in 1:length(lsk)
		matSkω[ik, iω] = 2 * integrate(lst, real.(matSkt[ik, :] .* lscoef))
	end
end
if LocalSpace == SU2Spin
     matSkω *= 2/3 
end

fig = Figure(size = (480, 300))
ax = Axis(fig[1, 1];
     xlabel = L"k / \pi", 
     ylabel = L"\omega",
     limits = ((0, 2), extrema(lsω))
)

hm = heatmap!(ax, lsk, lsω, matSkω)
Colorbar(fig[1, 2], hm; label = L"S^{+-}(k, \omega)")

display(fig)