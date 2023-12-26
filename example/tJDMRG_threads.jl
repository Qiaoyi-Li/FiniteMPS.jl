using MKL
using FiniteMPS, FiniteLattices

verbose = 1

include("Models/tJ.jl")

# show julia nthreads (set by -t)
@show Threads.nthreads()
@assert Threads.nthreads() > 1

@show TensorKit.Strided.set_num_threads(1)
# set MKL nthreads
BLAS.set_num_threads(1)
@show BLAS.get_num_threads()

flush(stdout)

@show Latt = SquaLatt(8, 4; BCY=:PBC)
disk = false # store local tensors in disk or memory
GCstep = false
Ψ = nothing

function mainDMRG(Ψ=nothing)

    Para = (t=1, t′=0.2, J=1//3)
    Ndop = 4 # number of hole doping, negative value means elec doping

    # =============== list D ====================
    lsD = broadcast(Int64 ∘ round, 2 .^ vcat(6:12))
    Nsweep = 1
    lsD = repeat(lsD, inner=Nsweep)
    # ===========================================

    # finish with 1-DMRG 
    Nsweep_DMRG1 = 2

    lsEn = zeros(length(lsD))
    lsinfo = Vector{Any}(undef, length(lsD))
    lsTimer = Vector{Any}(undef, length(lsD))

    # initial state
    if isnothing(Ψ)
        aspace = vcat(Rep[U₁×SU₂]((Ndop, 0) => 1), repeat([Rep[U₁×SU₂]((i, j) => 2 for i in -(abs(Ndop) + 1):(abs(Ndop)+1) for j in 0:1//2:1),], length(Latt) - 1))
        Ψ = randMPS(U₁SU₂tJFermion.pspace, aspace)
    end

    H = AutomataMPO(U₁SU₂tJ(Latt; Para...))
    Env = Environment(Ψ', H, Ψ; disk=disk)

    for (i, D) in enumerate(lsD)
        lsinfo[i], lsTimer[i] = DMRGSweep2!(Env;
            GCstep=GCstep, GCsweep=true, verbose=verbose,
            trunc=truncdim(D) & truncbelow(1e-6),
            LanczosOpt=(krylovdim=5, maxiter=1, tol=1e-4, orth=ModifiedGramSchmidt(), eager=true, verbosity=0))
        lsEn[i] = lsinfo[i][2][1].Eg
    end

    for i in 1:Nsweep_DMRG1
        info_DMRG1, Timer_DMRG1 = DMRGSweep1!(Env;
          CBEAlg=StandardCBE(lsD[end] + div(lsD[end], 10), 1e-6),
          trunc=truncdim(lsD[end]) & truncbelow(1e-6),
            GCstep=GCstep, GCsweep=true, verbose=verbose,
            LanczosOpt=(krylovdim=8, maxiter=1, tol=1e-4, orth=ModifiedGramSchmidt(), eager=true, verbosity=0))
        push!(lsEn, info_DMRG1[2].dmrg[1].Eg)
        push!(lsinfo, info_DMRG1)
        push!(lsTimer, Timer_DMRG1)
        flush(stdout)
    end

    GC.gc()
    return Ψ, Env, lsD, lsEn,  lsinfo, lsTimer
end

function mainObs(Ψ::MPS)

     Tree = ObservableTree()
     for i in 1:length(Latt)
          addObs!(Tree, U₁SU₂tJFermion.n, i, 1; name=:n)
     end
     for i in 1:length(Latt), j in i+1:length(Latt)
          addObs!(Tree, U₁SU₂tJFermion.SS, (i, j), 1; name=(:S, :S))
          addObs!(Tree, (U₁SU₂tJFermion.n, U₁SU₂tJFermion.n), (i, j), 1; name=(:n, :n))
          addObs!(Tree, U₁SU₂tJFermion.FdagF, (i, j), 1; Z=U₁SU₂tJFermion.Z, name=(:Fdag, :F))
     end
     for (i, j) in neighbor(Latt), (k, l) in neighbor(Latt)
          !isempty(intersect([i, j], [k, l])) && continue
          i > j && continue # note ⟨Δᵢⱼ^dag Δₖₗ⟩ = ⟨Δₖₗ^dag Δᵢⱼ⟩
          addObs!(Tree, U₁SU₂tJFermion.ΔₛdagΔₛ, (i, j, k, l), 1; Z=U₁SU₂tJFermion.Z, name=(:Fdag, :Fdag, :F, :F))
     end

     @time calObs!(Tree, Ψ; GCspacing=1000, verbose=verbose, showtimes=20)

     Obs = convert(Dict, Tree, [(:n,), (:nd,), (:S, :S), (:n, :n), (:Fdag, :F), (:Fdag, :Fdag, :F, :F)])
     GC.gc()

     return Obs

end

Ψ, Env, lsD, lsEn, lsinfo, lsTimer = mainDMRG(Ψ);
Obs = mainObs(Ψ)
lsEn
