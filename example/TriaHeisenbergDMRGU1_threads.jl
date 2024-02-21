using MKL
using FiniteMPS, FiniteLattices

verbose = 1

include("Models/Heisenberg.jl")

GCstep = false

@show Latt = YCTriangular(2, 3, SnakePath)
# SquaLatt(8, 4; BCY=:PBC)
disk = false # store local tensors in disk or memory

Ψ = nothing

function mainDMRG(Ψ=nothing)

     Para = (J=1, J′=0)

     lsD = let
          lsD = broadcast(Int64 ∘ round, 2 .^ vcat(3:10))
          Nsweep = 2
          repeat(lsD, inner=Nsweep)
     end
     lsnoise = zeros(length(lsD))
     # append!(lsnoise, zeros(length(lsD) - length(lsnoise)))

     # finish with 1-DMRG
     Nsweep_DMRG1 = 2

     lsEn = zeros(length(lsD))
     lsinfo = Vector{Any}(undef, length(lsD))
     lsTimer = Vector{Any}(undef, length(lsD))

     # initial state
     if isnothing(Ψ)
          aspace = vcat(Rep[U₁](0 => 1), repeat([Rep[U₁](j => 2 for j in -2//2:1//2:2//2),], length(Latt) - 1))
          Ψ = randMPS(U₁Spin.pspace, aspace)
     end

     H = AutomataMPO(U₁Heisenberg(Latt; Para...))
     Env = Environment(Ψ', H, Ψ; disk=disk)

     for (i, D) in enumerate(lsD)
          lsinfo[i], lsTimer[i] = DMRGSweep2!(Env;
               noise = lsnoise[i],
               GCstep=GCstep, GCsweep=true, verbose=verbose,
               trunc=truncdim(D) & truncbelow(1e-6),
               LanczosOpt=(krylovdim=5, maxiter=1, tol=1e-4, orth=ModifiedGramSchmidt(), eager=true, verbosity=0))
          lsEn[i] = lsinfo[i][2][1].Eg
     end

     for i in 1:Nsweep_DMRG1
          info_DMRG1, Timer_DMRG1 = DMRGSweep1!(Env;
               GCstep=false, GCsweep=true, verbose=verbose,
               LanczosOpt=(krylovdim=8, maxiter=1, tol=1e-4, orth=ModifiedGramSchmidt(), eager=true, verbosity=0))
          push!(lsEn, info_DMRG1[2].dmrg[1].Eg)
          push!(lsinfo, info_DMRG1)
          push!(lsTimer, Timer_DMRG1)
          flush(stdout)
     end

     GC.gc()
     return Ψ, Env, lsD, lsEn, lsinfo, lsTimer
end

function mainObs(Ψ::MPS)

     Tree = ObservableTree()
     for i in 1:length(Latt), j in i+1:length(Latt)
          addObs!(Tree, (U₁Spin.Sz, U₁Spin.Sz), (i, j), 1; name=(:Sz, :Sz))
          addObs!(Tree, U₁Spin.S₊₋, (i, j), 1; name=(:Sp, :Sm))
     end

     for i in 1:length(Latt)
          addObs!(Tree, U₁Spin.Sz, i, 1; name=:Sz)
     end

     @time calObs!(Tree, Ψ; GCspacing=1000, verbose=verbose, showtimes=20)

     Obs = convert(Dict, Tree, [(:Sz, :Sz), (:Sp, :Sm), (:Sz,)])
     GC.gc()

     return Obs

end

Ψ, Env, lsD, lsEn, lsinfo, lsTimer = mainDMRG(Ψ);
Obs = mainObs(Ψ)
lsEn
