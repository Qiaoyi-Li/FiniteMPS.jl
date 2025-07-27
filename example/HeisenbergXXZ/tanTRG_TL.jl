# Finite-T study of triangular lattice Heisenberg model   
# focus on YC geometry, SU2 and RSVD-CBE for acceleration
# plot energy, specific heat and uniform susceptibility

using FiniteMPS, FiniteLattices, CairoMakie, Statistics

include("Model.jl")
L = 6
W = 3
D = 1024
etol = 1e-8 # tolerance of per site energy

Latt = YCTria(L, W)

# Hamiltonian MPO 
H = AutomataMPO(HeisenbergXXZ(Latt; LocalSpace = SU2Spin))

# beta list, log scale before 1, linear scale after 1 
lsβ = vcat(2.0 .^ (-15:0), 2:16)
lsF = fill(NaN, length(lsβ))
lsE = fill(NaN, length(lsβ))
lsχ = fill(NaN, length(lsβ))

# SETTN initialization 
ρ, lsF_SETTN = SETTN(H, lsβ[1];
     CBEAlg = NaiveCBE(D + div(D, 4), 1e-8; rsvd = true),
	trunc = truncdim(256) & truncbelow(1e-16),
	maxorder = 4, verbose = 1, GCsweep = true,
	maxiter = 6, lsnoise = [0.1, 0.01, 0.001],
)
lsF[1] = lsF_SETTN[end]
lnZ = 2 * log(norm(ρ))
normalize!(ρ)

# generate the trilayer environment
Env = Environment(ρ', H, ρ)
lsE[1] = scalar!(Env)

# Observables
Obs = ObservableTree(size(Latt))
for i in 1:size(Latt), j in i:size(Latt)
	addObs!(Obs, SU2Spin.SS, (i, j), (false, false); name = (:S, :S))
end
calObs!(Obs, ρ)

# χ = \beta S(0)
lsχ[1] = lsβ[1] / 3 * sum([(i, j) for i in 1:size(Latt) for j in i:size(Latt)]) do (i, j)
	Sij = Obs.Refs["SS"][(i, j)][]
	return i == j ? Sij : 2 * Sij
end

# TDVP cooling
for idx in 2:length(lsβ)
	dβ = lsβ[idx] - lsβ[idx-1]

	TDVPSweep1!(Env, -dβ / 2;
		CBEAlg = NaiveCBE(D + div(D, 4), 1e-8; rsvd = true),
		GCsweep = true, verbose = 1,
		trunc = truncdim(D) & truncbelow(1e-12),
	)

	lnZ += 2 * log(norm(ρ))
	normalize!(ρ)

	lsF[idx] = -lnZ / lsβ[idx]
	lsE[idx] = scalar!(Env)

	# update Obs values 
	calObs!(Obs, ρ)
	lsχ[idx] = lsβ[idx] / 3 * sum([(i, j) for i in 1:size(Latt) for j in i:size(Latt)]) do (i, j)
		Sij = Obs.Refs["SS"][(i, j)][]
		return i == j ? Sij : 2 * Sij
	end

     @show lsβ[idx], lsF[idx], lsE[idx] 
end

# compute C = - ∂S / ∂lnβ
lsS = lsβ .* (lsE .- lsF)
lslnβ = log.(lsβ)
lsCe = - diff(lsS) ./ diff(lslnβ)
lsβ_c = exp.((lslnβ[1:end-1] + lslnβ[2:end])/2)

fig = Figure(size = (480, 600))
ax1 = Axis(fig[1, 1];
     ylabel = L"e",
     xscale = log10,
     limits = (0.05, 10, nothing, nothing),
)
ax2 = Axis(fig[2, 1];
     ylabel = L"c_v",
     xscale = log10,
     limits = (0.05, 10, 0, nothing),
)
ax3 = Axis(fig[3, 1];
     xlabel = L"T",
     ylabel = L"\chi",
     xscale = log10,
     limits = (0.05, 10, 0, nothing),
)
hidexdecorations!(ax1; grid = false, ticks = false)
hidexdecorations!(ax2; grid = false, ticks = false)
scatterlines!(ax1, 1 ./ lsβ, lsE ./ size(Latt))
scatterlines!(ax2, 1 ./ lsβ_c, lsCe ./ size(Latt))
scatterlines!(ax3, 1 ./ lsβ, lsχ ./ size(Latt))

display(fig)

