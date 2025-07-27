# Ground state study of square lattice AFM Heisenberg Model 
# use RSVD-CBE + noise for acceleration

using FiniteMPS, FiniteLattices, CairoMakie, Statistics

include("Model.jl")

LocalSpace = SU2Spin # NoSymSpinOneHalf, U1Spin, SU2Spin
L = 12
W = 6
J′ = 0.0
D = 256
etol = 1e-8 # tolerance of per site energy

# W * L cylinder
@assert iseven(L) && iseven(W) # so that (pi, pi) is valid
Latt = YCSqua(L, W)

# Hamiltonian MPO 
H = AutomataMPO(HeisenbergXXZ(Latt; LocalSpace = LocalSpace, J′ = J′))

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

# compute spin correlations 
Obs = ObservableTree(size(Latt))
for i in 1:size(Latt), j in i:size(Latt)
	if LocalSpace == SU2Spin
		addObs!(Obs, LocalSpace.SS, (i, j), (false, false); name = (:S, :S))
	else
		addObs!(Obs, (LocalSpace.Sz, LocalSpace.Sz), (i, j), (false, false); name = (:Sz, :Sz))
	end
end
calObs!(Obs, Ψ)

# unit = π
lskx = 0:2/L:1
lsky = 0:2/W:1
lsk_path = vcat(
	[(kx, 0.0) for kx in lskx][1:end-1], # (0, 0) -> (1, 0)
	[(1.0, ky) for ky in lsky][1:end-1], # (1, 0) -> (1, 1)
	[(k, k) for k in intersect(lskx, lsky)][end:-1:1] # (1, 1) -> (0, 0)
)
# generate the x axis
lsx = [0.0]
for i in 1:length(lsk_path)-1 
	push!(lsx, lsx[end] + sqrt(sum((lsk_path[i+1] .- lsk_path[i]) .^ 2)))
end

# compute the spin structure factor 
matS = zeros(size(Latt), size(Latt))
for i in 1:size(Latt), j in i:size(Latt)
	if LocalSpace == SU2Spin
		matS[i, j] = matS[j, i] = Obs.Refs["SS"][(i, j)][] / 3
	else 
		matS[i, j] = matS[j, i] = Obs.Refs["SzSz"][(i, j)][]
	end
end
# FT 
lsSk = map(lsk_path) do (kx, ky)
	FT2(matS, Latt, (kx*π, ky*π))
end

fig = Figure(size = (480, 600))
ax1 = Axis(fig[1, 1];
	xlabel = "nsweep",
	ylabel = L"e",
)
ax2 = Axis(fig[2, 1];
	ylabel = L"S(k)",
	xticks = ([0, 1, 2, 2+sqrt(2)], [L"(0, 0)", L"(\pi, 0)", L"(\pi, \pi)", L"(0, 0)"]),
	limits = (0, lsx[end], 0, nothing),
)
scatterlines!(ax1, lsE / L)
scatterlines!(ax2, lsx, lsSk)

display(fig)
