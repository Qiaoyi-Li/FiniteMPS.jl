function TDVPSweep2!(Env::SparseEnvironment{L,3,T}, dt::Number, direction::Symbol; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}

     @assert direction ∈ (:L, :R)
     trunc = get(kwargs, :trunc, truncbelow(MPSDefault.tol))
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)

     TimerSweep = TimerOutput()
     info_forward = Vector{TDVPInfo{2}}(undef, L - 1)
     info_backward = Vector{TDVPInfo{1}}(undef, L - 2)

     Ψ = Env[3]
     if direction == :L # left to right sweep
          Al::MPSTensor = Ψ[1]
          canonicalize!(Ψ, 1, 2)
          @timeit TimerSweep "TDVPSweep2>>" for si in 1:L-1
               TimerStep = TimerOutput()
               @timeit TimerStep "pushEnv" canonicalize!(Env, si, si + 1)
               Ar = Ψ[si+1]

               @timeit TimerStep "TDVPUpdate2" xg, Norm, info_Lanczos = _TDVPUpdate2(ProjHam(Env, si, si + 1), Al, Ar, dt; kwargs...)
               @timeit TimerStep "svd" Ψ[si], Al, info_svd = leftorth(xg; trunc=trunc)
               # remember to change Center of Ψ manually
               Center(Ψ)[:] = [si + 1, si + 1]
               rmul!(Ψ, Norm)
               info_forward[si] = TDVPInfo{2}(dt, info_Lanczos, info_svd)

               # backward evolution
               if si < L - 1
                    @timeit TimerStep "pushEnv" canonicalize!(Env, si + 1, si + 1)
                    @timeit TimerStep "TDVPUpdate1" Al, Norm, info_Lanczos = _TDVPUpdate1(ProjHam(Env, si + 1, si + 1), Al, -dt; kwargs...)
                    rmul!(Ψ, Norm)
                    info_backward[si] = TDVPInfo{1}(-dt, info_Lanczos, BondInfo(Al, :R))
               end
               # GC manually
               GCstep && manualGC(TimerStep)

               # show
               merge!(TimerSweep, TimerStep; tree_point=["TDVPSweep2>>"])
               if verbose ≥ 2
                    show(TimerStep; title="site $(si)->$(si+1)")
                    let K = info_forward[si].Lanczos.numops, info_svd = info_forward[si].Bond
                         println("\nForward evolution: K = $(K), $(info_svd)")
                    end
                    if si < L - 1
                         let K = info_backward[si].Lanczos.numops, info_svd = info_backward[si].Bond
                              println("Backward evolution: K = $(K), $(info_svd)")
                         end
                    end
                    flush(stdout)
               end

          end
          Ψ[L] = Al

     else # right to left sweep
          Ar::MPSTensor = Ψ[L]
          canonicalize!(Ψ, L - 1, L)
          @timeit TimerSweep "TDVPSweep2<<" for si = reverse(2:L)
               TimerStep = TimerOutput()
               @timeit TimerStep "pushEnv" canonicalize!(Env, si - 1, si)
               Al = Ψ[si-1]

               @timeit TimerStep "TDVPUpdate2" xg, Norm, info_Lanczos = _TDVPUpdate2(ProjHam(Env, si - 1, si), Al, Ar, dt; kwargs...)
               @timeit TimerStep "svd" Ar, Ψ[si], info_svd = rightorth(xg; trunc=trunc)
               # remember to change Center of Ψ manually
               Center(Ψ)[:] = [si - 1, si - 1]
               rmul!(Ψ, Norm)
               info_forward[si-1] = TDVPInfo{2}(dt, info_Lanczos, info_svd)

               # backward evolution
               if si > 2
                    @timeit TimerStep "pushEnv" canonicalize!(Env, si - 1, si - 1)
                    @timeit TimerStep "TDVPUpdate1" Ar, Norm, info_Lanczos = _TDVPUpdate1(ProjHam(Env, si - 1, si - 1), Ar, -dt; kwargs...)
                    rmul!(Ψ, Norm)
                    info_backward[si-2] = TDVPInfo{1}(-dt, info_Lanczos, BondInfo(Ar, :L))
               end

               # GC manually
               GCstep && manualGC(TimerStep)

               # show
               merge!(TimerSweep, TimerStep; tree_point=["TDVPSweep2<<"])
               if verbose ≥ 2
                    show(TimerStep; title="site $(si-1)<-$(si)")
                    let K = info_forward[si-1].Lanczos.numops, info_svd = info_forward[si-1].Bond
                         println("\nForward evolution: K = $(K), $(info_svd)")
                    end
                    if si > 2
                         let K = info_backward[si-2].Lanczos.numops, info_svd = info_backward[si-2].Bond
                              println("Backward evolution: K = $(K), $(info_svd)")
                         end
                    end
                    flush(stdout)
               end
          end
          Ψ[1] = Ar
     end

     # GC manually
     GCsweep && manualGC()

     return (forward=info_forward, backward=info_backward), TimerSweep

end

function TDVPSweep2!(Env::SparseEnvironment{L,3,T}, dt::Number; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     # sweep left to right and then right to left symmetrically
     verbose::Int64 = get(kwargs, :verbose, 0)
     nsweep = get(kwargs, :nsweep, "")
     info = map((:L, :R)) do direction
          info, TimerSweep = TDVPSweep2!(Env, dt / 2, direction; kwargs...)
          if verbose ≥ 1
               show(TimerSweep; title="TDVP sweep $(nsweep) $(direction)")
               let K = maximum(x -> x.Lanczos.numops, info.forward)
                    bondinfo_merge = merge(map(x -> x.Bond, info.forward))
                    println("\nForward evolution: K = $(K), $(bondinfo_merge)")
               end
               let K = maximum(x -> x.Lanczos.numops, info.backward)
                    bondinfo_merge = merge(map(x -> x.Bond, info.backward))
                    println("Backward evolution: K = $(K), $(bondinfo_merge)")
               end
               flush(stdout)
          end
          info
     end
     return info
end

function _TDVPUpdate2(H::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor, dt::Number; kwargs...)

     expx, info = _LanczosExp(x -> action2(H, x; kwargs...),
          dt,
          CompositeMPSTensor(Al, Ar),
          _getLanczos(; kwargs...))

     # normalize
     Norm = norm(expx)
     rmul!(expx, 1 / Norm)
     return expx, Norm, info

end

function _TDVPUpdate1(H::SparseProjectiveHamiltonian{1}, A::MPSTensor, dt::Number; kwargs...)

     expx, info = _LanczosExp(x -> action1(H, x; kwargs...),
          dt,
          A,
          _getLanczos(; kwargs...))

     # normalize
     Norm = norm(expx)
     rmul!(expx, 1 / Norm)
     return expx, Norm, info

end

function _LanczosExp(f, t::Number, x, args...; kwargs...)
     # wrap KrylovKit.exponentiate to make sure the integrate step length is exactly t

     x, info = exponentiate(f, t, x, args...; kwargs...)
     residual = info.residual
     K_sum = info.numops
     while residual != 0
          dt = sign(t)*residual
          x, info = exponentiate(f, dt, x, args...; kwargs...)
          @show residual = info.residual
          K_sum += info.numops
     end
     info = LanczosInfo(info.converged > 0, [info.normres,], info.numiter, K_sum)
     return x, info
end