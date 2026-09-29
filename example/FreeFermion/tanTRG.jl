using FiniteMPS, FiniteLattices, CairoMakie, BenchmarkFreeFermions
import FiniteMPS: update!
using LinearAlgebra

include("Model.jl")
L = 6
W = 2
t = 0.1
μ = 0.0
β₀ = 2.0 ^ (-15)
βm = 8.0 / t
D = 512

opt = AdaptiveWidthOptimizer(1.0, 10.0)
# opt = AdaptiveGradientOptimizer(1.0, 10.0, 1.0)

Latt = YCSqua(L, W)
lsβ = Float64[]
lslnZ = Float64[]
lsE = Float64[]

# Hamiltonian MPO 
H = AutomataMPO(SpinlessFreeFermion(Latt; t = t, μ = μ))

# SETTN initialization
DSETTN = 256 
ρ, _ = SETTN(H, β₀;
     CBEAlg = NaiveCBE(DSETTN + div(DSETTN, 4), 1e-8; rsvd = true),
	trunc = truncrank(DSETTN) & trunctol(; atol = 1e-16),
	maxorder = 4, verbose = 1, GCsweep = true,
	maxiter = 6, lsnoise = [(1/4, x) for x in [0.1, 0.01, 0.001]],
)
lnZ = 2 * log(norm(ρ))
normalize!(ρ)

# generate the trilayer environment
Env = Environment(ρ', H, ρ)

push!(lsβ, β₀)
push!(lslnZ, lnZ)
push!(lsE, scalar!(Env))

# status, ∇e2 and W are irrelevant when β is small
st = ImaginaryTimeStatus(lsβ[end], 1.0, 1.0)

# TDVP cooling
while lsβ[end] < βm 
     dβ = opt(st)
     info, _ = TDVPSweep1!(Env, -dβ / 2;
		CBEAlg = NaiveCBE(D + div(D, 4), 1e-8; rsvd = true),
		GCsweep = true, verbose = 1,
		trunc = truncrank(D) & trunctol(; atol = 1e-12),
	)

     # estimate spectral width of local effective Hamiltonian
     W = maximum(info[2].forward) do info_i 
          T_i = diagm(0 => info_i.Lanczos.a, -1 => info_i.Lanczos.b, 1 => info_i.Lanczos.b)
          ϵ_i = eigvals(T_i) 
          return ϵ_i[end] - ϵ_i[1]
     end 
     ∇E2 = maximum(info[2].forward) do info_i  
          return 4 * info_i.Lanczos.b[1]^2 # <H^2> = a1^2 + b1^2, <H> = a1, note ∇E = 2(H - E)|Ψ⟩    
     end
     update!(st, lsβ[end] + dβ, ∇E2 / size(Latt), W)


     lnZ += 2 * log(norm(ρ))
	normalize!(ρ)

     push!(lsβ, lsβ[end] + dβ)
     push!(lslnZ, lnZ)
	push!(lsE, scalar!(Env))

     @show st
     @show lslnZ[end], lsE[end] 
     

end

# exact solution 
Tij = zeros(Float64, size(Latt), size(Latt))
for (i, j) in neighbor(Latt; ordered = true)
     Tij[i, j] = t 
end
ϵ = SingleParticleSpectrum(Tij)

lslnZ_ex = map(lsβ) do β 
     LogPartition(ϵ, β, μ)
end
lsE_ex = map(lsβ) do β 
     Energy(ϵ, β, μ)
end

fig = Figure(size = (480, 600))
ax1 = Axis(fig[1, 1];
     xlabel = L"\beta",
     ylabel = L"\delta\beta",
     limits = (0, βm, 0, nothing),
)
ax2 = Axis(fig[2, 1];
     xlabel = L"\beta",
     ylabel = L"\ln Z",
     limits = (0, βm, nothing, nothing),
)
ax3 = Axis(fig[3, 1];
     xlabel = L"\beta",
     ylabel = L"E",
     limits = (0, βm, nothing, nothing),
)
hidexdecorations!(ax1; grid = false, ticks = false)
hidexdecorations!(ax2; grid = false, ticks = false)

scatterlines!(ax1, lsβ[1:end-1], diff(lsβ))

lines!(ax2, lsβ, lslnZ; label = "tanTRG")
scatter!(ax2, lsβ, lslnZ_ex; color = :red, label = "exact", markersize = 4)
axislegend(ax2; position = (0.0, 1.0))

lines!(ax3, lsβ, lsE; label = "tanTRG")
scatter!(ax3, lsβ, lsE_ex; color = :red, label = "exact", markersize = 4)

display(fig)
