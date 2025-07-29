"""
     DMRGSweep2!(Env::SparseEnvironment{L,3,T}, ::SweepDirection; kwargs...) 
          -> info::Vector{DMRGInfo}, Timer::TimerOutput

2-site DMRG sweep from left to right or sweep back from right to left.  

# Kwargs
     K::Int64 = 16
Krylov space dimension.

     tol::Real = 1e-8
Tolerance for eagerly break in Lanczos iteration.

     trunc::TruncationScheme = truncbelow(MPSDefault.tol) & truncdim(MPSDefault.D)
Control the truncation in svd after each 2-site update. Details see `tsvd`. 

     GCstep::Bool = false
`GC.gc()` manually after each step if `true`.

     GCsweep::Bool = false 
`GC.gc()` manually after each (left to right or right to left) sweep if `true`.  

     verbose::Int64 = 0
Print the `TimerOutput` after each sweep or each local update if `verbose = 1` or `2`, respectively.

     noise::Real = 0
Add noise to the 2-site local tensor after each update via applying a random gate to the physical indices. Note this is only a naive implementation, and may not work well for some cases. 
"""
function DMRGSweep2!(Env::SparseEnvironment{L,3,T}, ::SweepL2R; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}
     @assert Center(Env[3])[2] ≤ 2

     K = get(kwargs, :K, 16)
     tol = get(kwargs, :tol, 1e-8)
     trunc = get(kwargs, :trunc, truncbelow(MPSDefault.tol) & truncdim(MPSDefault.D))
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     noise::Float64 = get(kwargs, :noise, 0)
     @assert noise ≥ 0

     TimerSweep = TimerOutput()
     info = Vector{DMRGInfo}(undef, L - 1)

     Ψ = Env[3]
     canonicalize!(Ψ, 1, 2)
     Al::MPSTensor = Ψ[1]
     # shift energy
     E₀::Float64 = scalar!(Env; normalize=true) |> real
     @timeit TimerSweep "DMRGSweep2>>" for si = 1:L-1
          TimerStep = TimerOutput()
          @timeit TimerStep "pushEnv" canonicalize!(Env, si, si + 1)
          Ar = Ψ[si+1]

          PH = CompositeProjectiveHamiltonian(Env.El[si], Env.Er[si+1], (Env[2][si], Env[2][si+1]), E₀)
          @timeit TimerStep "DMRGUpdate2" eg, xg, info_Lanczos = LanczosGS(action, CompositeMPSTensor(Al, Ar), PH, TimerStep;
               K = K, tol = tol, verbose = false)
          finalize(PH)

          eg += E₀
          # apply noise
          if noise > 0 && si < L-1
               noise!(xg, noise)
          end
          @timeit TimerStep "svd" Ψ[si], s, vd, info_svd = tsvd(xg; trunc=trunc)
          # next Al
          Al = s * vd
          # remember to change Center of Ψ manually
          Center(Ψ)[:] = [si + 1, si + 1]
          info[si] = DMRGInfo(eg, info_Lanczos, info_svd)

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep2>>"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si)->$(si+1)")
               let K = info[si].Lanczos.numops
                    println("\nK = $(K), $(info_svd), Eg = $(eg)")
               end
               flush(stdout)
          end

     end
     Ψ[L] = normalize!(Al)

     # GC manually
     GCsweep && manualGC(TimerSweep)

     return info, TimerSweep

end

function DMRGSweep2!(Env::SparseEnvironment{L,3,T}, ::SweepR2L; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}
     @assert Center(Env[3])[1] ≥ L - 1

     K = get(kwargs, :K, 16)
     tol = get(kwargs, :tol, 1e-8)
     trunc = get(kwargs, :trunc, truncbelow(MPSDefault.tol) & truncdim(MPSDefault.D))
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     noise::Float64 = get(kwargs, :noise, 0)
     @assert noise ≥ 0

     TimerSweep = TimerOutput()
     info = Vector{DMRGInfo}(undef, L - 1)

     Ψ = Env[3]
     canonicalize!(Ψ, L - 1, L)
     E₀::Float64 = scalar!(Env; normalize=true) |> real
     Ar::MPSTensor = Ψ[L]
     @timeit TimerSweep "DMRGSweep2<<" for si = reverse(2:L)
          TimerStep = TimerOutput()
          @timeit TimerStep "pushEnv" canonicalize!(Env, si - 1, si)
          Al = Ψ[si-1]

          PH = CompositeProjectiveHamiltonian(Env.El[si-1], Env.Er[si], (Env[2][si-1], Env[2][si]), E₀)
          @timeit TimerStep "DMRGUpdate2" eg, xg, info_Lanczos = LanczosGS(action, CompositeMPSTensor(Al, Ar), PH, TimerStep;
               K = K, tol = tol, verbose = false)
          finalize(PH)

          eg += E₀
          # apply noise
          # TODO: try to mix phys and bond idx
          if noise > 0 && si > 2
               noise!(xg, noise)
          end
          @timeit TimerStep "svd" u, s, Ψ[si], info_svd = tsvd(xg; trunc=trunc)
          # next Ar
          Ar = u * s
          # remember to change Center of Ψ manually
          Center(Ψ)[:] = [si - 1, si - 1]

          info[si-1] = DMRGInfo(eg, info_Lanczos, info_svd)

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep2<<"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si-1)<-$(si)")
               let K = info[si-1].Lanczos.numops
                    println("\nK = $(K), $(info_svd), Eg = $(eg)")
               end
               flush(stdout)
          end
     end
     Ψ[1] = normalize!(Ar)

     # GC manually
     GCsweep && manualGC(TimerSweep)

     return info, TimerSweep

end

"""
     DMRGSweep1!(Env::SparseEnvironment{L,3,T}, ::SweepDirection; kwargs...)
          -> info::Vector{DMRGInfo}, Timer::TimerOutput

1-site DMRG sweep from left to right or sweep back from right to left.  

# Kwargs
     K::Int64 = 16
Krylov space dimension.

     tol::Real = 1e-8
Tolerance for eagerly break in Lanczos iteration.

     GCstep::Bool = false
`GC.gc()` manually after each step if `true`.

     GCsweep::Bool = false 
`GC.gc()` manually after each (left to right or right to left) sweep if `true`.   

     verbose::Int64 = 0
Print the `TimerOutput` after each sweep or each local update if `verbose = 1` or `2`, respectively.

     CBEAlg::CBEAlgorithm = NoCBE()
CBE algorithm for 1-DMRG.

     trunc::TruncationScheme = notrunc()
Control the truncation after each update, only used together with CBE. Details see `tsvd`. 

     noise::NTuple{2, Float64} = (0.1, 0.0)
Add noise to the 1-site local tensor after each Lanczos update via expanding the bond and add a random tensor (normal distribution) to it. The first element is the ratio of additional bond dimension `(≤ 1.0)`, the second element is the noise strength.
"""
function DMRGSweep1!(Env::SparseEnvironment{L,3,T}, ::SweepL2R; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}

     K = get(kwargs, :K, 16)
     tol = get(kwargs, :tol, 1e-8)
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     CBEAlg::CBEAlgorithm{SweepL2R} = get(kwargs, :CBEAlg, NoCBE())
     trunc = get(kwargs, :trunc, notrunc())
     noise::NTuple{2, Float64} = get(kwargs, :noise, (0.1, 0.0))
     @assert noise[1] ≤ 1.0 && noise[2] ≥ 0.0

     TimerSweep = TimerOutput()
     info_dmrg = Vector{DMRGInfo}(undef, L)
     info_cbe = Vector{Union{Nothing,CBEInfo}}(nothing, L - 1)

     Ψ = Env[3]
     canonicalize!(Ψ, 1)
     canonicalize!(Env, 1)
     # shift energy
     E₀::Float64 = scalar!(Env; normalize=true) |> real
     Al::MPSTensor = Ψ[1]
     @timeit TimerSweep "DMRGSweep1>>" for si = 1:L
          TimerStep = TimerOutput()

          # CBE 
          if !isa(CBEAlg, NoCBE) && si < L
               canonicalize!(Env, si, si + 1)
               @timeit TimerStep "CBE" Al, Ψ[si+1], info_cbe[si], TO_CBE  = CBE(Al, Ψ[si+1], Env.El[si], Env.Er[si+1], Env[2][si], Env[2][si+1], CBEAlg)
               merge!(TimerStep, TO_CBE; tree_point=["CBE"])
          end

          @timeit TimerStep "pushEnv" canonicalize!(Env, si)
          PH = CompositeProjectiveHamiltonian(Env.El[si], Env.Er[si], (Env[2][si],), E₀)
          @timeit TimerStep "DMRGUpdate1" eg, xg, info_Lanczos = LanczosGS(action, Al, PH, TimerStep;
               K = K, tol = tol, verbose = false)
          finalize(PH)

          eg += E₀
          if si < L
               # add noise 
               @timeit TimerStep "noise" if noise[2] > 0 
                    De = ceil(Int64, noise[1]*dim(codomain(xg)))
                    Ve = _rsvd_trunc(codomain(xg), De)
                    A_noise = randn(eltype(xg), codomain(xg), Ve)
                    rmul!(A_noise, noise[2] / norm(A_noise))
                    xg = MPSTensor(catdomain(xg.A, A_noise))
                    normalize!(xg)
               
                    Ar = permute(Ψ[si+1].A, ((1,), (2, 3)))
                    Ar_0 = zeros(eltype(Ar), Ve, domain(Ar))
                    Ψ[si + 1] = catcodomain(Ar, Ar_0)
               end

               @timeit TimerStep "svd" Ψ[si], S::MPSTensor, info_svd = leftorth(xg; trunc=trunc)
               # next Al
               Al = S * Ψ[si+1]
               (noise[2] > 0) && normalize!(Al)
               # remember to change Center of Ψ manually
               Center(Ψ)[:] = [si + 1, si + 1]
          else
               Ψ[si] = normalize!(xg)
               info_svd = BondInfo(xg, :R)
          end
          info_dmrg[si] = DMRGInfo(eg, info_Lanczos, info_svd)

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep1>>"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si)->")
               let K = info_dmrg[si].Lanczos.numops
                    println("\nK = $(K), $(info_svd), Eg = $(eg)")
               end
               flush(stdout)
          end
     end

     # GC manually
     GCsweep && manualGC(TimerSweep)

     return (dmrg=info_dmrg, cbe=info_cbe), TimerSweep

end


function DMRGSweep1!(Env::SparseEnvironment{L,3,T}, ::SweepR2L; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}

     K = get(kwargs, :K, 16)
     tol = get(kwargs, :tol, 1e-8)
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     CBEAlg::CBEAlgorithm{SweepR2L} = get(kwargs, :CBEAlg, NoCBE())
     trunc = get(kwargs, :trunc, notrunc())
     noise::NTuple{2, Float64} = get(kwargs, :noise, (0.1, 0.0))
     @assert noise[1] ≤ 1.0 && noise[2] ≥ 0.0

     TimerSweep = TimerOutput()
     info_dmrg = Vector{DMRGInfo}(undef, L)
     info_cbe = Vector{Union{Nothing,CBEInfo}}(nothing, L - 1)

     Ψ = Env[3]
     canonicalize!(Ψ, L)
     canonicalize!(Env, L)
     # shift energy
     E₀::Float64 = scalar!(Env; normalize=true) |> real
     Ar::MPSTensor = Ψ[L]
     @timeit TimerSweep "DMRGSweep1<<" for si = reverse(1:L)
          TimerStep = TimerOutput()

          # CBE
          if !isa(CBEAlg, NoCBE) && si > 1
               canonicalize!(Env, si - 1, si)
               @timeit TimerStep "CBE" Ψ[si-1], Ar, info_cbe[si-1], TO_CBE = CBE(Ψ[si-1], Ar, Env.El[si-1], Env.Er[si], Env[2][si-1], Env[2][si],  CBEAlg)
               merge!(TimerStep, TO_CBE; tree_point=["CBE"])
          end

          @timeit TimerStep "pushEnv" canonicalize!(Env, si)
          PH = CompositeProjectiveHamiltonian(Env.El[si], Env.Er[si], (Env[2][si],), E₀)
          @timeit TimerStep "DMRGUpdate1" eg, xg, info_Lanczos = LanczosGS(action, Ar, PH, TimerStep;
               K = K, tol = tol, verbose = false)
          finalize(PH)
          eg += E₀
          if si > 1
               # add noise 
               @timeit TimerStep "noise" if noise[2] > 0 
                    Ar_p = permute(xg.A, ((1,), (2, 3)))
                    De = ceil(Int64, noise[1]*dim(domain(Ar_p)))
                    Ve = _rsvd_trunc(domain(Ar_p), De)

                    A_noise = randn(eltype(xg), Ve, domain(Ar_p))
                    rmul!(A_noise, noise[2] / norm(A_noise))
                    xg = MPSTensor(catcodomain(Ar_p, A_noise))
                    normalize!(xg)
               
                    Al = Ψ[si - 1].A
                    Al_0 = zeros(eltype(Al), codomain(Al), Ve)
                    Ψ[si - 1] = catdomain(Al, Al_0)
               end


               @timeit TimerStep "svd" S::MPSTensor, Ψ[si], info_svd = rightorth(xg; trunc=trunc)
               # next Ar
               Ar = Ψ[si-1] * S
               (noise[2] > 0) && normalize!(Ar)
               # remember to change Center of Ψ manually
               Center(Ψ)[:] = [si - 1, si - 1]
          else
               Ψ[si] = normalize!(xg)
               info_svd = BondInfo(xg, :L)
          end
          info_dmrg[si] = DMRGInfo(eg, info_Lanczos, info_svd)

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep1<<"])
          if verbose ≥ 2
               show(TimerStep; title="site <-$(si)")
               let K = info_dmrg[si].Lanczos.numops
                    println("\nK = $(K), $(info_svd), Eg = $(eg)")
               end
               flush(stdout)
          end

     end

     # GC manually
     GCsweep && manualGC(TimerSweep)

     return (dmrg=info_dmrg, cbe=info_cbe), TimerSweep

end

for func in (:DMRGSweep1!, :DMRGSweep2!)
     @eval begin
          function $func(Env::SparseEnvironment{L,3,T}; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}

               verbose::Int64 = get(kwargs, :verbose, 0)
               lsinfo = Vector{Any}(undef, 2)
               lsTimer = Vector{TimerOutput}(undef, 2)
               for (i, direction) in enumerate((SweepL2R(), SweepR2L()))
                    lsinfo[i], lsTimer[i] = $func(Env, direction; kwargs...)
                    if verbose ≥ 1
                         str = i == 1 ? ">>" : "<<"
                         show(lsTimer[i]; title="DMRG sweep $(str)")
                         info_dmrg = try
                              lsinfo[i].dmrg # CBE 1-DMRG
                         catch
                              lsinfo[i] # 2-DMRG 
                         end
                         let K = maximum(x -> x.Lanczos.numops, info_dmrg), Eg = info_dmrg[i == 1 ? end : 1].Eg
                              bondinfo_merge = merge(map(x -> x.Bond, info_dmrg))
                              println("\nK = $(K), $(bondinfo_merge), Eg = $(Eg)")
                         end
                         flush(stdout)
                    end
               end
               return lsinfo, lsTimer
          end

     end
end

