"""
     TDVPSweep1!(Env::SparseEnvironment{L,3,T},
          dt::Number,
          direction::Symbol;
          kwargs...)

Apply left to right (`direction = :L`) or right to left (`direction = :R`) 1-site TDVP`[https://doi.org/10.1103/PhysRevB.94.165116]` sweep to perform time evolution for `DenseMPS` (both MPS and MPO) with step length `dt`. `Env` is the 3-layer environment `⟨Ψ|H|Ψ⟩`.
     
     TDVPSweep1!(Env::SparseEnvironment{L,3,T}, dt::Number; kwargs...)
Wrap `TDVPSweep1!` with a symmetric integrator, i.e., sweeping from left to right and then from right to left with the same step length `dt / 2`.

# Kwargs
     trunc::TruncationType = truncbelow(MPSDefault.tol)
     GCstep::Bool = false
     GCsweep::Bool = false
     verbose::Int64 = 0
     LanczosOpt::NameTuple (Pack options of Lanczos algorithm, details please see `KrylovKit.jl`)
"""
function TDVPSweep1!(Env::SparseEnvironment{L,3,T}, dt::Number, direction::Symbol; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}

     @assert direction ∈ (:L, :R)
     trunc = get(kwargs, :trunc, truncbelow(MPSDefault.tol))
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)

     TimerSweep = TimerOutput()
     info_forward = Vector{TDVPInfo{1}}(undef, L)
     info_backward = Vector{TDVPInfo{0}}(undef, L - 1)

     Ψ = Env[3]
     # shift energy
     E₀::Float64 = scalar!(Env; normalize=true)
     if direction == :L # left to right sweep
          canonicalize!(Ψ, 1)
          Al::MPSTensor = Ψ[1]
          @timeit TimerSweep "TDVPSweep1>>" for si in 1:L
               TimerStep = TimerOutput()
               @timeit TimerStep "pushEnv" canonicalize!(Env, si)

               @timeit TimerStep "TDVPUpdate1" x1, Norm, info_Lanczos = _TDVPUpdate1(ProjHam(Env, si; E₀=E₀), Al, dt; kwargs...)
               rmul!(Ψ, Norm * exp(real(dt) * E₀))

               if si < L
                    @timeit TimerStep "svd" Ψ[si], S::MPSTensor, info_svd = leftorth(x1; trunc=trunc)
                    # note svd may change the norm of S
                    normalize!(S)
                    info_forward[si] = TDVPInfo{1}(dt, info_Lanczos, info_svd)

                    # backward evolution
                    @timeit TimerStep "pushEnv" canonicalize!(Env, si + 1, si)
                    @timeit TimerStep "TDVPUpdate0" S, Norm, info_Lanczos = _TDVPUpdate0(ProjHam(Env, si + 1, si; E₀=E₀), S, -dt; kwargs...)
                    rmul!(Ψ, Norm * exp(-real(dt) * E₀))

                    # next Al
                    Al = S * Ψ[si+1]
                    # remember to change Center of Ψ manually
                    Center(Ψ)[:] = [si + 1, si + 1]

                    info_backward[si] = TDVPInfo{0}(-dt, info_Lanczos, BondInfo(Al, :L))

                    # update E₀, note Norm ~ exp(-dt * (⟨H⟩ - E₀))
                    E₀ -= log(Norm) / real(dt)
               else
                    Ψ[si] = x1
                    info_forward[si] = TDVPInfo{1}(dt, info_Lanczos, BondInfo(x1, :R))
               end

               # GC manually
               GCstep && manualGC(TimerStep)

               # show
               merge!(TimerStep, get_timer("action1"); tree_point=["TDVPUpdate1"])
               si < L && merge!(TimerStep, get_timer("action0"); tree_point=["TDVPUpdate0"])
               merge!(TimerSweep, TimerStep; tree_point=["TDVPSweep1>>"])
               if verbose ≥ 2
                    show(TimerStep; title="site $(si)->")
                    let K = info_forward[si].Lanczos.numops, info_svd = info_forward[si].Bond
                         println("\nForward: K = $(K), $(info_svd)")
                    end
                    if si < L
                         let K = info_backward[si].Lanczos.numops, info_svd = info_backward[si].Bond
                              println("Backward: K = $(K), $(info_svd)")
                         end
                    end
                    flush(stdout)
               end

          end

     else # right to left sweep
          canonicalize!(Ψ, L)
          Ar::MPSTensor = Ψ[L]
          @timeit TimerSweep "TDVPSweep1<<" for si = reverse(1:L)
               TimerStep = TimerOutput()
               @timeit TimerStep "pushEnv" canonicalize!(Env, si)

               @timeit TimerStep "TDVPUpdate1" x1, Norm, info_Lanczos = _TDVPUpdate1(ProjHam(Env, si; E₀=E₀), Ar, dt; kwargs...)
               rmul!(Ψ, Norm * exp(real(dt) * E₀))

               if si > 1
                    @timeit TimerStep "svd" S::MPSTensor, Ψ[si], info_svd = rightorth(x1; trunc=trunc)
                    # note svd may change the norm of S
                    normalize!(S)
                    info_forward[si] = TDVPInfo{1}(dt, info_Lanczos, info_svd)

                    # backward evolution
                    @timeit TimerStep "pushEnv" canonicalize!(Env, si, si - 1)
                    @timeit TimerStep "TDVPUpdate0" S, Norm, info_Lanczos = _TDVPUpdate0(ProjHam(Env, si, si - 1; E₀=E₀), S, -dt; kwargs...)
                    rmul!(Ψ, Norm * exp(-real(dt) * E₀))

                    # next Ar
                    Ar = Ψ[si-1] * S
                    # remember to change Center of Ψ manually
                    Center(Ψ)[:] = [si - 1, si - 1]

                    info_backward[si-1] = TDVPInfo{0}(-dt, info_Lanczos, BondInfo(Ar, :R))

                    # update E₀, note Norm ~ exp(-dt * (⟨H⟩ - E₀))
                    E₀ -= log(Norm) / real(dt)

               else
                    Ψ[si] = x1
                    info_forward[si] = TDVPInfo{1}(dt, info_Lanczos, BondInfo(x1, :L))
               end

               # GC manually
               GCstep && manualGC(TimerStep)

               # show
               merge!(TimerStep, get_timer("action1"); tree_point=["TDVPUpdate1"])
               si > 1 && merge!(TimerStep, get_timer("action0"); tree_point=["TDVPUpdate0"])
               merge!(TimerSweep, TimerStep; tree_point=["TDVPSweep1<<"])
               if verbose ≥ 2
                    show(TimerStep; title="site <-$(si)")
                    let K = info_forward[si].Lanczos.numops, info_svd = info_forward[si].Bond
                         println("\nForward: K = $(K), $(info_svd)")
                    end
                    if si > 1
                         let K = info_backward[si-1].Lanczos.numops, info_svd = info_backward[si-1].Bond
                              println("Backward: K = $(K), $(info_svd)")
                         end
                    end
                    flush(stdout)
               end
          end
     end

     # GC manually
     GCsweep && manualGC()

     return (forward=info_forward, backward=info_backward), TimerSweep

end

function TDVPSweep1!(Env::SparseEnvironment{L,3,T}, dt::Number; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     # sweep left to right and then right to left symmetrically
     verbose::Int64 = get(kwargs, :verbose, 0)
     info = map((:L, :R)) do direction
          info, TimerSweep = TDVPSweep1!(Env, dt / 2, direction; kwargs...)
          if verbose ≥ 1
               str = direction == :L ? ">>" : "<<"
               show(TimerSweep; title="TDVP sweep $(str)")
               let K = maximum(x -> x.Lanczos.numops, info.forward)
                    bondinfo_merge = merge(map(x -> x.Bond, info.forward))
                    println("\nForward: K = $(K), $(bondinfo_merge)")
               end
               let K = maximum(x -> x.Lanczos.numops, info.backward)
                    bondinfo_merge = merge(map(x -> x.Bond, info.backward))
                    println("Backward: K = $(K), $(bondinfo_merge)")
               end
               flush(stdout)
          end
          info
     end
     return info
end

