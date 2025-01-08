using FiniteMPS, FiniteLattices

include("model.jl")
verbose = 1
GCstep = true
Latt = YCSqua(8, 4) |> Zigzag!
t′ = 0.0
U = 8
Ndop = 0 # number of hole doping
Qspin = 0 # SU2 spin, half integer
@assert Qspin % 1 == (iseven(Ndop) ? 0 : 1//2) 

lsD = let
     lsD = broadcast(Int64 ∘ round, 2 .^ vcat(6:12))
     repeat(lsD, inner=2)
end
lsE = zeros(length(lsD))
lsinfo = Vector{Any}(undef, length(lsD))
lsTimer = Vector{Any}(undef, length(lsD))

# initial state
Ψ = let
     aspace = vcat(Rep[U₁×SU₂]((Ndop, Qspin) => 1), repeat([Rep[U₁×SU₂]((i, j) => 1 for i in -(abs(Ndop) + 1):(abs(Ndop)+1) for j in 0:1//2:1),], size(Latt) - 1))
     randMPS(U₁SU₂Fermion.pspace, aspace)
end

let
     H = AutomataMPO(U₁SU₂Hubbard(Latt; t′ = t′, U = U))
     Env = Environment(Ψ', H, Ψ)

     for (i, D) in enumerate(lsD)
          lsinfo[i], lsTimer[i] = DMRGSweep2!(Env;
               GCstep=GCstep, GCsweep=true, verbose=verbose,
               trunc=truncdim(D) & truncbelow(1e-8))
          lsE[i] = lsinfo[i][2][1].Eg
     end

     GC.gc()
end

# Observables
Obs = let 
     Tree = ObservableTree()
     for i in 1:size(Latt)
          addObs!(Tree, U₁SU₂Fermion.n, i, 1; name=:n)
     end
     for i in 1:size(Latt), j in i+1:size(Latt)
          addObs!(Tree, U₁SU₂Fermion.SS, (i, j), (false, false); name=(:S, :S))
          addObs!(Tree, (U₁SU₂Fermion.n, U₁SU₂Fermion.n), (i, j), (false, false); name=(:n, :n))
          addObs!(Tree, U₁SU₂Fermion.FdagF, (i, j), (true, true); Z=U₁SU₂Fermion.Z, name=(:Fdag, :F))
     end

     for (i, j) in neighbor(Latt), (k, l) in neighbor(Latt)
          !isempty(intersect([i, j], [k, l])) && continue
          (i > j || k > l) && continue # note Δᵢⱼ = Δⱼᵢ
          i > k && continue  # note ⟨Δᵢⱼ^dag Δₖₗ⟩ = ⟨Δₖₗ^dag Δᵢⱼ⟩

          # singlet pairing
          addObs!(Tree, U₁SU₂Fermion.ΔₛdagΔₛ, (i, j, k, l), Tuple(fill(true, 4)); Z=U₁SU₂Fermion.Z, name=(:Fdag, :FdagS, :FS, :F))
          # triplet pairing
          addObs!(Tree, U₁SU₂Fermion.ΔₜdagΔₜ, (i, j, k, l), Tuple(fill(true, 4)); Z=U₁SU₂Fermion.Z, name=(:Fdag, :FdagT, :FT, :F))
     end

     @time calObs!(Tree, Ψ; GCspacing=1000, verbose=verbose, showtimes=10)

     convert(Dict, Tree)
end

