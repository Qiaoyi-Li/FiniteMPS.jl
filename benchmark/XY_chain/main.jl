using FiniteMPS
using CairoMakie
using BenchmarkFreeFermions
using LinearAlgebra

"""
     This script provides a benchmark between linearly increasing beta list and exponentially increasing beta list (in the initial stage), which is proposed in our tanTRG paper (https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.226502).

"""


include("model.jl")

# XY chain 
# convention: J(SxSx + SySy)
L = 32
J = 1.0
D = 64

lsβ_lin = 0.2:0.2:8.0
lsβ_exp2lin = vcat(2.0 .^ (-10:1:0), 2.0:1.0:8.0)
lsF_lin = ExactSolution(L, lsβ_lin; J = J)
lsF_exp2lin = ExactSolution(L, lsβ_exp2lin; J = J)

H = AutomataMPO(XYChain(L; J = J))

# 1. linearly increasing β from Id
lsF_1 = let
	ρ = identityMPO(L, U1Spin.pspace)
	lnZ = 2 * log(norm(ρ))
	normalize!(ρ)
	Env = Environment(ρ', H, ρ)
	canonicalize!(Env, 1)
	map(1:length(lsβ_lin)) do i
		dβ = i == 1 ? lsβ_lin[1] : lsβ_lin[i] - lsβ_lin[i-1]

		TDVPSweep2!(Env, -dβ / 2;
			trunc = truncdim(D))
		lnZ += 2 * log(norm(ρ))
		normalize!(ρ)

		F = -lnZ / lsβ_lin[i]
		println("β = $(lsβ_lin[i]), F = $(F), δF = $(abs(F - lsF_lin[i]))")
		return F
	end
end

# 2. exponentially increasing β in initial stage
lsF_2 = let
	ρ = identityMPO(L, U1Spin.pspace)
	lnZ = 2 * log(norm(ρ))
	normalize!(ρ)
	Env = Environment(ρ', H, ρ)
	canonicalize!(Env, 1)
	map(1:length(lsβ_exp2lin)) do i
		dβ = i == 1 ? lsβ_exp2lin[1] : lsβ_exp2lin[i] - lsβ_exp2lin[i-1]

		TDVPSweep2!(Env, -dβ / 2;
			trunc = truncdim(D))
		lnZ += 2 * log(norm(ρ))
		normalize!(ρ)

		F = -lnZ / lsβ_exp2lin[i]
		println("β = $(lsβ_exp2lin[i]), F = $(F), δF = $(abs(F - lsF_exp2lin[i]))")
		return F
	end
end

# 3. initialize high-T state via SETTN
lsF_3 = let
	ρ, _ = SETTN(H, lsβ_exp2lin[1];
		maxorder = 4, tol = 1e-16,
		compress = 0.0,
		trunc = truncdim(D),
	)
	lnZ = 2 * log(norm(ρ))
	normalize!(ρ)
	Env = Environment(ρ', H, ρ)
	canonicalize!(Env, 1)
	map(1:length(lsβ_exp2lin)) do i
		i == 1 && return -lnZ / lsβ_exp2lin[1]
		dβ = lsβ_exp2lin[i] - lsβ_exp2lin[i-1]

		TDVPSweep2!(Env, -dβ / 2;
			trunc = truncdim(D))
		lnZ += 2 * log(norm(ρ))
		normalize!(ρ)

		F = -lnZ / lsβ_exp2lin[i]
		println("β = $(lsβ_exp2lin[i]), F = $(F), δF = $(abs(F - lsF_exp2lin[i]))")
		return F
	end
end


# plot
fig = Figure(size = (400, 300))
ax = Axis(fig[1, 1];
	xlabel = L"\beta",
	ylabel = L"\delta F",
	yscale = log10,
)

lines!(ax, lsβ_lin, abs.(lsF_1 - lsF_lin), color = :blue, label = "linear")
lines!(ax, lsβ_exp2lin, abs.(lsF_2 - lsF_exp2lin), color = :red, label = "exp2lin")
lines!(ax, lsβ_exp2lin, abs.(lsF_3 - lsF_exp2lin), color = :green, label = "exp2lin + SETTN")


axislegend(ax, position = (1.0, 0.0))

display(fig)