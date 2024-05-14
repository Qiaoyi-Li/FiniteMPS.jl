"""
     DMRGSweep2!(Env::SparseEnvironment{L,3,T}, ::SweepDirection; kwargs...) 
          -> info::Vector{DMRGInfo}, Timer::TimerOutput

2-site DMRG sweep from left to right or sweep back from right to left.  

# Kwargs
     krylovalg::KrylovKit.KrylovAlgorithm = DMRGDefaultLanczos
Krylov algorithm used in DMRG update.

     trunc::TruncationScheme = truncbelow(MPSDefault.tol) & truncdim(MPSDefault.D)
Control the truncation in svd after each 2-site update. Details see `tsvd`. 

     GCstep::Bool = false
`GC.gc()` manually after each step if `true`.

     GCsweep::Bool = false 
`GC.gc()` manually after each (left to right or right to left) sweep if `true`.  

     verbose::Int64 = 0
Print the `TimerOutput` after each sweep or each local update if `verbose = 1` or `2`, respectively.

     noise::Real = 0
Add noise to the 2-site local tensor after each update. 
"""
function DMRGSweep2!(Env::SparseEnvironment{L,3,T}, ::SweepL2R; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}
     @assert Center(Env[3])[2] ≤ 2

     krylovalg = get(kwargs, :krylovalg, DMRGDefaultLanczos)
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

          @timeit TimerStep "DMRGUpdate2" eg, xg, info_Lanczos = _DMRGUpdate2(ProjHam(Env, si, si + 1; E₀=E₀), Al, Ar, krylovalg; kwargs...)
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
          merge!(TimerStep, get_timer("action2"); tree_point=["DMRGUpdate2"])
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep2>>"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si)->$(si+1)")
               let K = info_Lanczos.numops
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

     krylovalg = get(kwargs, :krylovalg, DMRGDefaultLanczos)
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

          @timeit TimerStep "DMRGUpdate2" eg, xg, info_Lanczos = _DMRGUpdate2(ProjHam(Env, si - 1, si; E₀=E₀), Al, Ar, krylovalg; kwargs...)
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
          merge!(TimerStep, get_timer("action2"); tree_point=["DMRGUpdate2"])
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep2<<"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si-1)<-$(si)")
               let K = info_Lanczos.numops
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
     krylovalg::KrylovKit.KrylovAlgorithm = DMRGDefaultLanczos
Krylov algorithm used in DMRG update.

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
"""
function DMRGSweep1!(Env::SparseEnvironment{L,3,T}, ::SweepL2R; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}

     krylovalg = get(kwargs, :krylovalg, DMRGDefaultLanczos)
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     CBEAlg::CBEAlgorithm{SweepL2R} = get(kwargs, :CBEAlg, NoCBE())
     trunc = get(kwargs, :trunc, notrunc())

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
               @timeit TimerStep "CBE" Al, Ψ[si+1], info_cbe[si] = CBE(ProjHam(Env, si, si + 1; E₀=E₀), Al, Ψ[si+1], CBEAlg)
               merge!(TimerStep, get_timer("CBE"); tree_point=["CBE"])
          end

          @timeit TimerStep "pushEnv" canonicalize!(Env, si)
          @timeit TimerStep "DMRGUpdate1" eg, xg, info_Lanczos = _DMRGUpdate1(ProjHam(Env, si; E₀=E₀), Al, krylovalg; kwargs...)
          eg += E₀
          if si < L
               @timeit TimerStep "svd" Ψ[si], S::MPSTensor, info_svd = leftorth(xg; trunc=trunc)
               # next Al
               Al = S * Ψ[si+1]
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
          merge!(TimerStep, get_timer("action1"); tree_point=["DMRGUpdate1"])
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep1>>"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si)->")
               let K = info_Lanczos.numops
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

     krylovalg = get(kwargs, :krylovalg, DMRGDefaultLanczos)
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     CBEAlg::CBEAlgorithm{SweepR2L} = get(kwargs, :CBEAlg, NoCBE())
     trunc = get(kwargs, :trunc, notrunc())

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
               @timeit TimerStep "CBE" Ψ[si-1], Ar, info_cbe[si-1] = CBE(ProjHam(Env, si - 1, si; E₀=E₀), Ψ[si-1], Ar, CBEAlg)
               merge!(TimerStep, get_timer("CBE"); tree_point=["CBE"])
          end

          @timeit TimerStep "pushEnv" canonicalize!(Env, si)
          @timeit TimerStep "DMRGUpdate1" eg, xg, info_Lanczos = _DMRGUpdate1(ProjHam(Env, si; E₀=E₀), Ar, krylovalg; kwargs...)
          eg += E₀
          if si > 1
               @timeit TimerStep "svd" S::MPSTensor, Ψ[si], info_svd = rightorth(xg; trunc=trunc)
               # next Ar
               Ar = Ψ[si-1] * S
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
          merge!(TimerStep, get_timer("action1"); tree_point=["DMRGUpdate1"])
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep1<<"])
          if verbose ≥ 2
               show(TimerStep; title="site <-$(si)")
               let K = info_Lanczos.numops
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

function _DMRGUpdate2(H::SparseProjectiveHamiltonian{2},
     Al::MPSTensor,
     Ar::MPSTensor,
     alg::KrylovKit.KrylovAlgorithm;
     kwargs...)

     reset_timer!(get_timer("action2"))
     eg, xg, info = eigsolve(x -> action2(H, x; kwargs...), CompositeMPSTensor(Al, Ar), 1, :SR, alg)

     return eg[1], xg[1], info

end

function _DMRGUpdate1(H::SparseProjectiveHamiltonian{1},
     A::MPSTensor,
     alg::KrylovKit.KrylovAlgorithm;
     kwargs...)

     reset_timer!(get_timer("action1"))
     eg, xg, info = eigsolve(x -> action1(H, x; kwargs...), A, 1, :SR, alg)
     return eg[1], xg[1], info

end