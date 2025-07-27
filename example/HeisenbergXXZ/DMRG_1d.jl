# Ground state study of 1D AFM Heisenberg Model 

using FiniteMPS, FiniteLattices, CairoMakie, Statistics

include("Model.jl")

LocalSpace = NoSymSpinOneHalf # NoSymSpinOneHalf, U1Spin, SU2Spin
L = 64
J′ = 0.0
D = 128
etol = 1e-8 # tolerance of per site energy

# generate a 1d chain lattice, as a special case of square lattice
Latt = OpenSqua(L, 1)

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

# 2-DMRG sweep
for nsweep in 1:20
	info, _ = DMRGSweep2!(Env;
		verbose = 1, GCsweep = true,
		trunc = truncdim(D) & truncbelow(1e-12),
		K = 16)
	push!(lsE, info[2][1].Eg)

	if lsE[end-1] - lsE[end] < etol * size(Latt)
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

lsr = 1:L-1
lsSr = map(lsr) do r
	mean(1:L-r) do i
           # extract data 
          if LocalSpace == SU2Spin 
		Obs.Refs["SS"][(i, i + r)][] / 3
          else 
               Obs.Refs["SzSz"][(i, i + r)][]
          end
	end
end

fig = Figure(size = (480, 600))
ax1 = Axis(fig[1, 1];
	xlabel = "nsweep",
	ylabel = L"e",
)
ax2 = Axis(fig[2, 1];
	xlabel = L"r",
	ylabel = L"(-1)^r S(r)",
	xscale = log10,
	yscale = log10,
	limits = (1, L, nothing, nothing),
)
scatterlines!(ax1, lsE / L)
scatterlines!(ax2, lsr, (-1) .^ lsr .* lsSr)

display(fig)
