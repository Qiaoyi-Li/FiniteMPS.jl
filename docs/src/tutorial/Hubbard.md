# [Hubbard Model](@id Hubbard)

## Ground state
In this section, we use Hubbard model as an example to demonstrate how to simulate fermionic systems and compute multi-site correlations. We consider the square-lattice Hubbard model on a $W\times L$ cylinder. Note we use a relatively small bond dimension here to avoid too much online build time, which is not sufficient to obtain converged results.

The Hamiltonian of Hubbard model reads\
$H = -t\sum_{\langle i, j\rangle\sigma} c_{i\sigma}^\dagger c_{j\sigma} + U\sum_i n_{i\uparrow}n_{i\downarrow}$. Here we set $t = 1$ as energy unit and use charge U(1) and spin SU(2) symmetries. 
```@example Hubbard 
using FiniteMPS
using FiniteLattices
using CairoMakie, Statistics


mkpath("figs_Hubbard") # save figures

# parameters
L = 24
W = 4
U = 12.0
Ntot = 88 # total particle number
Dmin = 128 # min bond dimension
Dmax = 1024 # max bond dimension

# generate lattices with FiniteLattices.jl
Latt = YCSqua(L, W) |> Snake!

# generate the Hamiltonian MPO
Tree = InteractionTree(size(Latt))
# hopping
for (i, j) in neighbor(Latt; ordered = true)
     addIntr!(Tree, U1SU2Fermion.FdagF, (i, j), (true, true), -1.0; name = (:Fdag, :F), Z = U1SU2Fermion.Z)
end

# U terms
for i in 1:size(Latt)
     addIntr!(Tree, U1SU2Fermion.nd, i, U; name = :nd)
end
H = AutomataMPO(Tree)
```
The above code generates the Hamiltonian MPO for the Hubbard model, the only new usage is setting `(true, true)` to indicate both operators are fermionic and passing the parity operator `Z` when adding hopping terms. Next we generate a random density distribution and obtain the corresponding MPS.
```@example Hubbard 
# random density distribution
lsn = zeros(Int64, size(Latt))
for _ in 1:Ntot 
     i = rand(findall(==(minimum(lsn)), lsn))
     lsn[i] += 1
end

# charge quantum numbers of horizontal bonds
lsqc = [0] 
for n in reverse(lsn) # from right to left
     if n == 0
          push!(lsqc, lsqc[end] + 1)
     elseif n == 1
          push!(lsqc, lsqc[end])
     else
          push!(lsqc, lsqc[end] - 1)
     end
end
lsqc = reverse(lsqc[2:end])

# spaces of horizontal bonds
lsspaces = map(1:size(Latt)) do i 
     # the left boundary bond
     i == 1 && return Rep[U₁×SU₂]((lsqc[1], iseven(Ntot) ? 0 : 1/2) => 1)
     return Rep[U₁×SU₂]((lsqc[i], s) => 1 for s in 0:1/2:1/2)
end
Ψ = randMPS(fill(U1SU2Fermion.pspace, size(Latt)), lsspaces)
```
Then we prepare the `ObservableTree` used to calculate observables.
```@example Hubbard 
ObsTree = ObservableTree(size(Latt))

# on-site terms
for i in 1:size(Latt)
     addObs!(ObsTree, U1SU2Fermion.n, i; name = :n) # density
     addObs!(ObsTree, U1SU2Fermion.nd, i; name = :nd) # double occupancy
end

# all-to all 2-site correlations
for i in 1:size(Latt), j in i:size(Latt)
     # spin correlation
     addObs!(ObsTree, U1SU2Fermion.SS, (i, j), (false, false); name = (:S, :S))
     # density correlation
     addObs!(ObsTree, (U1SU2Fermion.n, U1SU2Fermion.n), (i, j), (false, false); name = (:n, :n))
     # single particle correlation
     addObs!(ObsTree, U1SU2Fermion.FdagF, (i, j), (true, true); name = (:Fdag, :F), Z = U1SU2Fermion.Z)
end

# singlet pairing correlations of y-bonds
Pairs = [(Latt[x, y], Latt[x, y % W + 1]) for x in 1:L for y in 1:W]
for idx1 in 1:length(Pairs), idx2 in idx1:length(Pairs)
     addObs!(ObsTree, U1SU2Fermion.ΔₛdagΔₛ, (Pairs[idx1]..., Pairs[idx2]...), (true, true, true, true); name = (:Fdag, :Fdag, :F, :F), Z = U1SU2Fermion.Z)
end

# observables of the initial random state
calObs!(ObsTree, Ψ)
Obs = convert(Dict, ObsTree)
```
We can verify that the initial density distribution matches the random configuration via
```@example Hubbard
# check initial density distribution
@assert all(i -> Obs["n"][(i,)] ≈ lsn[i], 1:length(lsn))
```
Then, we perform the CBE-DMRG to obtain the ground state.
```@example Hubbard
Env = Environment(Ψ', H, Ψ)
lsD = [Dmin]
lsE = Float64[] 
lsObs = Dict[]
etol = 1e-5 # energy tolerance
MaxSweeps = 100 # avoid dead loops

for i in 1:MaxSweeps
     D = lsD[end]
     info, _ = DMRGSweep1!(Env; K = 16, trunc = truncdim(D),
          CBEAlg = NaiveCBE(D + div(D, 4), 1e-8; rsvd = true),
     )
    
	push!(lsE, info[2].dmrg[1].Eg)
     # @show D, lsE[end]

     if i > 1 && (lsE[end-1] - lsE[end])/size(Latt) < etol 
          calObs!(ObsTree, Ψ)
          push!(lsObs, convert(Dict, ObsTree))
         
          D *= 2  # increase bond dimension
          D > Dmax && break # converged
     end

     push!(lsD, D)
end

# plot the energy vs nsweep
fig = Figure(size = (480, 240))
ax = Axis(fig[1, 1];
     xlabel = "nsweep",
     ylabel = "energy per site")
scatterlines!(ax, lsE / size(Latt)) # per site
save("figs_Hubbard/GS_energy.png", fig)
```
![](./figs_Hubbard/GS_energy.png)
The parameters are chosen as $U = 12$ and $\delta = 1/12$, where the ground state belongs to [LE1](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.033073) phase that exhibits half-filled charge density wave with $\lambda_\textrm{CDW}= 1/(2\delta) = 6$ and power-law pairing correlation.

```@example Hubbard
# illustrate CDW
fig = Figure(size = (480, 500))
ax = Axis(fig[1, 1];
     xlabel = L"x",
     ylabel = L"n(x)")
map(lsObs, unique(lsD), range(0.2, 1.0;length = length(lsObs))) do Obs, D, α
     lsnx = map(1:L) do x 
          return mean(1:W) do y 
               Obs["n"][(Latt[x, y],)]
          end
     end
     scatterlines!(ax, 1:L, lsnx;
          color = (:blue, α),
          label = L"D = %$(D)"
     )
end

# pairing correlation in 1/4 to 3/4 bulk region
ax2 = Axis(fig[2, 1];
     xlabel = L"r",
     ylabel = L"\Phi_{yy}(r)",
     xscale = log10,
     yscale = log10,
)
lsr = 1:div(L, 2) - 1
map(lsObs, unique(lsD), range(0.2, 1.0;length = length(lsObs))) do Obs, D, α
     lsΦyy = map(lsr) do r 
          mean([(x, y) for x in div(L, 4)+1 : div(3*L, 4)-r for y in 1:W]) do (x, y)
               Obs["FdagFdagFF"][(Latt[x, y], Latt[x, y % W + 1], Latt[x + r, y], Latt[x + r, y % W + 1])]
          end
     end
     scatterlines!(ax2, lsr, lsΦyy;
          color = (:blue, α),
          label = L"D = %$(D)"
     )
end
axislegend(ax2; position = (0, 0), rowgap = 0)

save("figs_Hubbard/GS_Obs.png", fig)
```
![](./figs_Hubbard/GS_Obs.png)
Finally, we illustrate the CDW and the pairing correlation. Emphasize again this is only a demonstrative example, a much larger bond dimension is required to reproduce the power-law pairing correlation.

## Finite temperature
TODO
