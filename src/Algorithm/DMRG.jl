"""
     DMRGSweep2!(Env::SparseEnvironment{L,3,T}; kwargs...)

2-site DMRG sweep from left to right and sweep back from right to left.  

# Kwargs
     trunc::TruncationScheme = notrunc()
Control the truncation in svd after each 2-site update. Details see `tsvd`. 

     GCstep::Bool = false
`GC.gc()` manually after each step if `true`.

     GCsweep::Bool = false 
`GC.gc()` manually after each (left to right or right to left) sweep if `true`.  

     verbose::Int64 = 0
Print the `TimerOutput` after each sweep or each local update if `verbose = 1` or `2`, respectively.
"""
function DMRGSweep2!(Env::SparseEnvironment{L,3,T}; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}

     trunc = get(kwargs, :trunc, notrunc())
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     TimerSweep = TimerOutput()
     # info, (L, R)
     info = (Vector{DMRGInfo}(undef, L - 1), Vector{DMRGInfo}(undef, L - 1))

     Ψ = Env[3]
     canonicalize!(Ψ, 1, 2)
     # left to right sweep
     Al::MPSTensor = Ψ[1]
     @timeit TimerSweep "DMRGSweep2>>" for si = 1:L-1
          TimerStep = TimerOutput()
          @timeit TimerStep "pushEnv" canonicalize!(Env, si, si + 1)
          Ar = Ψ[si+1]

          @timeit TimerStep "DMRGUpdate2" eg, xg, info_Lanczos = _DMRGUpdate2(ProjHam(Env, si, si + 1), Al, Ar; kwargs...)
          @timeit TimerStep "svd" Ψ[si], s, vd, info_svd = tsvd(xg; trunc=trunc)
          # next Al
          Al = s * vd
          # remember to change Center of Ψ manually
          Center(Ψ)[:] = [si + 1, si + 1]
          info[1][si] = DMRGInfo(eg, info_Lanczos, info_svd)

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep2>>"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si)->$(si+1)")
               let K = info_Lanczos.numops
                    println("\nK = $(K), $(info_svd), Eg = $(eg)")
               end
               flush(stdout)
          end

     end
     Ψ[L] = Al

     # GC manually
     GCsweep && manualGC()

     # right to left sweep
     Ar::MPSTensor = Al
     @timeit TimerSweep "DMRGSweep2<<" for si = reverse(2:L)
          TimerStep = TimerOutput()
          @timeit TimerStep "pushEnv" canonicalize!(Env, si - 1, si)
          Al = Ψ[si-1]

          @timeit TimerStep "DMRGUpdate2" eg, xg, info_Lanczos = _DMRGUpdate2(ProjHam(Env, si - 1, si), Al, Ar; kwargs...)
          @timeit TimerStep "svd" u, s, Ψ[si], info_svd = tsvd(xg; trunc=trunc)
          # next Ar
          Ar = u * s
          # remember to change Center of Ψ manually
          Center(Ψ)[:] = [si - 1, si - 1]

          info[2][si-1] = DMRGInfo(eg, info_Lanczos, info_svd)

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep2<<"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si-1)<-$(si)")
               let K = info_Lanczos.numops
                    println("\nK = $(K), $(info_svd), Eg = $(eg)")
               end
               flush(stdout)
          end
     end
     Ψ[1] = Ar

     # GC manually
     GCsweep && manualGC()
     global GlobalCountDMRGSweep2 += 1

     if verbose ≥ 1
          show(TimerSweep; title="sweep $(GlobalCountDMRGSweep2)")
          let K = maximum(x -> x.Lanczos.numops, info[2]), Eg = info[2][1].Eg
               bondinfo_merge = merge(map(x -> x.Bond, info[2]))
               println("\nK = $(K), $(bondinfo_merge), Eg = $(Eg)")
          end
          flush(stdout)
     end

     return info

end

"""
     DMRGSweep1!(Env::SparseEnvironment{L,3,T}; kwargs...)

1-site DMRG sweep from left to right and sweep back from right to left.  

# Kwargs
     GCstep::Bool = false
`GC.gc()` manually after each step if `true`.

     GCsweep::Bool = false 
`GC.gc()` manually after each (left to right or right to left) sweep if `true`.   

     verbose::Int64 = 0
Print the `TimerOutput` after each sweep or each local update if `verbose = 1` or `2`, respectively.
"""
function DMRGSweep1!(Env::SparseEnvironment{L,3,T}; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}

     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     TimerSweep = TimerOutput()
     # info, (L, R)
     info = (Vector{DMRGInfo}(undef, L), Vector{DMRGInfo}(undef, L))

     Ψ = Env[3]
     # left to right sweep
     @timeit TimerSweep "DMRGSweep1>>" for si = 1:L
          TimerStep = TimerOutput()
          @timeit TimerStep "canonicalize" canonicalize!(Ψ, si)
          @timeit TimerStep "pushEnv" canonicalize!(Env, si)

          @timeit TimerStep "DMRGUpdate1" eg, xg, info_Lanczos = _DMRGUpdate1(ProjHam(Env, si), Ψ[si]; kwargs...)
          Ψ[si] = xg
          info[1][si] = DMRGInfo(eg, info_Lanczos, BondInfo(xg, :R))

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep1>>"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si)->")
               let K = info_Lanczos.numops
                    println("\nK = $(K), $(info[1][si].Bond), Eg = $(eg)")
               end
               flush(stdout)
          end
     end

     # GC manually
     GCsweep && manualGC()

     # right to left sweep
     @timeit TimerSweep "DMRGSweep1<<" for si = reverse(1:L)
          TimerStep = TimerOutput()
          @timeit TimerStep "canonicalize" canonicalize!(Ψ, si)
          @timeit TimerStep "pushEnv" canonicalize!(Env, si)

          @timeit TimerStep "DMRGUpdate1" eg, xg, info_Lanczos = _DMRGUpdate1(ProjHam(Env, si), Ψ[si]; kwargs...)
          Ψ[si] = xg
          info[2][si] = DMRGInfo(eg, info_Lanczos, BondInfo(xg, :L))

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
          merge!(TimerSweep, TimerStep; tree_point=["DMRGSweep1<<"])
          if verbose ≥ 2
               show(TimerStep; title="site <-$(si)")
               let K = info_Lanczos.numops
                    println("\nK = $(K), $(info[2][si].Bond), Eg = $(eg)")
               end
               flush(stdout)
          end

     end

     # GC manually
     GCsweep && manualGC()
     global GlobalCountDMRGSweep1 += 1

     if verbose ≥ 1
          show(TimerSweep; title="sweep $(GlobalCountDMRGSweep1)")
          let K = maximum(x -> x.Lanczos.numops, info[2]), Eg = info[2][1].Eg
               bondinfo_merge = merge(map(x -> x.Bond, info[2]))
               println("\nK = $(K), $(bondinfo_merge), Eg = $(Eg)")
          end
          flush(stdout)
     end

     return info

end

function _DMRGUpdate2(H::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor; kwargs...)

     eg, xg, info = eigsolve(x -> action2(H, x; kwargs...), CompositeMPSTensor(Al, Ar), 1, :SR, _getLanczos(; kwargs...))

     return eg[1], xg[1], info

end

function _DMRGUpdate1(H::SparseProjectiveHamiltonian{1}, A::MPSTensor; kwargs...)

     eg, xg, info = eigsolve(x -> action1(H, x; kwargs...), A, 1, :SR, _getLanczos(; kwargs...))
     return eg[1], xg[1], info

end