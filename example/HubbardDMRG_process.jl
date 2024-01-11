using Distributed
using FiniteLattices

NWORKERS = 4 # number of workers
NTHREADS = 2 # number of threads of subworkers, suggestion = 1
verbose = 1

addprocs(NWORKERS, exeflags=["--threads=$(NTHREADS)"])
@show Distributed.workers()

@everywhere begin
     using MKL, FiniteMPS
     @show FiniteMPS.get_num_threads_julia()
     @show BLAS.get_config()
     BLAS.set_num_threads(1) # set MKL nthreads, suggestion = 1
     @show BLAS.get_num_threads()
end

flush(stdout)

include("Models/Hubbard.jl")

@show Latt = SquaLatt(8, 4; BCY=:PBC)
disk = false # store local tensors in disk or memory

Ψ = nothing

function mainDMRG(Ψ=nothing)

     Para = (t=1, t′=-0.2, U=8)
     Ndop = 0 # number of hole doping, negative value means elec doping

     # =============== list D ====================
     lsD = broadcast(Int64 ∘ round, 2 .^ vcat(6:12))
     Nsweep = 2
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
          Ψ = randMPS(U₁SU₂Fermion.pspace, aspace; disk=disk)
     end

     H = AutomataMPO(U₁SU₂Hubbard(Latt; Para...))
     Env = Environment(Ψ', H, Ψ; disk=disk)


     for (i, D) in enumerate(lsD)
          lsinfo[i], lsTimer[i] = DMRGSweep2!(Env;
               GCstep=true, GCsweep=true, verbose = verbose,
               trunc=truncdim(D) & truncbelow(1e-6),
               LanczosOpt=(krylovdim=5, maxiter=1, tol=1e-4, orth=ModifiedGramSchmidt(), eager=true, verbosity=0))
          lsEn[i] = lsinfo[i][2][1].Eg
     end

     for i in 1:Nsweep_DMRG1
          info_DMRG1, Timer_DMRG1 = DMRGSweep1!(Env;
               GCstep=false, GCsweep=true, verbose = verbose,
               LanczosOpt=(krylovdim=8, maxiter=1, tol=1e-4, orth=ModifiedGramSchmidt(), eager=true, verbosity=0))
          push!(lsEn, info_DMRG1[2].dmrg[1].Eg)
          push!(lsinfo, info_DMRG1)
          push!(lsTimer, Timer_DMRG1)
          flush(stdout)

     end

     GC.gc()
     return Ψ, Env, lsD, lsEn, lsinfo, lsTimer
end

Ψ, Env, lsD, lsEn, lsinfo, lsTimer = mainDMRG(Ψ);

rmprocs(workers())
