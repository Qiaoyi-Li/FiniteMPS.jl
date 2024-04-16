using FiniteMPS, FiniteLattices

FiniteMPS.set_num_threads_mkl(1)

include("model.jl")
verbose = 1
GCstep = false
Latt = YCTria(9, 6) |> Snake! # Triangular lattice
# Latt = YCSqua(8, 4) |> Snake! # Square lattice
J′ = 0.0

lsD = let
     lsD = broadcast(Int64 ∘ round, 2 .^ vcat(6:12))
     repeat(lsD, inner=2)
end
lsE = zeros(length(lsD))
lsinfo = Vector{Any}(undef, length(lsD))
lsTimer = Vector{Any}(undef, length(lsD))

# initial state
Ψ = let
     aspace = vcat(Rep[SU₂](0 => 1), repeat([Rep[SU₂](i => 1 for i in 0:1//2:1),], size(Latt) - 1))
     randMPS(SU₂Spin.pspace, aspace)
end

let
     H = AutomataMPO(SU₂Heisenberg(Latt; J′ = J′))
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
     for i in 1:size(Latt), j in i+1:size(Latt)
          addObs!(Tree, SU₂Spin.SS, (i, j), 1; name=(:S, :S))
     end

     @time calObs!(Tree, Ψ; GCspacing=1000, verbose=verbose, showtimes=10)

     convert(Dict, Tree, [(:S, :S),])
end

