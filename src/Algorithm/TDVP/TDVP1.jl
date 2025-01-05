"""
     TDVPSweep1!(Env::SparseEnvironment{L,3,T},
          dt::Number,
          direction::SweepDirection;
          kwargs...) -> info, TimerSweep

Apply 1-site TDVP`[https://doi.org/10.1103/PhysRevB.94.165116]` sweep to perform time evolution for `DenseMPS` (both MPS and MPO) with step length `dt`. `Env` is the 3-layer environment `⟨Ψ|H|Ψ⟩`.
     
     TDVPSweep1!(Env::SparseEnvironment{L,3,T}, dt::Number; kwargs...)
Wrap `TDVPSweep1!` with a symmetric integrator, i.e., sweeping from left to right and then from right to left with the same step length `dt / 2`.

# Kwargs
     K::Int64 = 32 
     tol::Float64 = 1e-8
The maximum Krylov dimension and tolerance in Lanczos exponential method.

     trunc::TruncationType = notrunc()
     GCstep::Bool = false
     GCsweep::Bool = false
     verbose::Int64 = 0
     CBEAlg::CBEAlgorithm = NoCBE()
     E_shift::Float64 = 0.0
Apply `exp(dt(H - E_shift))` to avoid possible `Inf` in imaginary time evolution. This energy shift is different from `E₀` in projective Hamiltonian, the later will give back the shifted energy thus not altering the final result. Note this is a temporary approach, we intend to store `log(norm)` in MPS to avoid this divergence in the future.
"""
function TDVPSweep1!(Env::SparseEnvironment{L,3,T}, dt::Number, direction::SweepL2R; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     K = get(kwargs, :K, 32)
     tol = get(kwargs, :tol, 1e-8)
     trunc = get(kwargs, :trunc, notrunc())
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     CBEAlg::CBEAlgorithm{SweepL2R} = get(kwargs, :CBEAlg, NoCBE())
     E_shift::Float64 = get(kwargs, :E_shift, 0.0)

     TimerSweep = TimerOutput()
     info_forward = Vector{TDVPInfo{1}}(undef, L)
     info_backward = Vector{TDVPInfo{0}}(undef, L - 1)
     info_cbe = Vector{Union{Nothing, CBEInfo}}(nothing, L - 1)

     Ψ = Env[3]
     canonicalize!(Ψ, 1)
     canonicalize!(Env, 1)
     # shift energy
     E₀::Float64 = scalar!(Env; normalize=true) |> real
     Al::MPSTensor = Ψ[1]
     @timeit TimerSweep "TDVPSweep1>>" for si in 1:L
          TimerStep = TimerOutput()

          # CBE
          if !isa(CBEAlg, NoCBE) && si < L
               canonicalize!(Env, si, si + 1)
               @timeit TimerStep "CBE" Al, Ψ[si+1], info_cbe[si], TO_CBE = CBE(Al, Ψ[si+1], Env.El[si], Env.Er[si+1], Env[2][si], Env[2][si+1], CBEAlg)
               merge!(TimerStep, TO_CBE; tree_point=["CBE"])
          end

          @timeit TimerStep "pushEnv" canonicalize!(Env, si)
          PH = CompositeProjectiveHamiltonian(Env.El[si], Env.Er[si], (Env[2][si],), E₀)
          @timeit TimerStep "TDVPUpdate1" x1, info_Lanczos = LanczosExp(action, Al, dt, PH, TimerStep; K=K, tol=tol, verbose=false)
          finalize(PH)
          Norm = norm(x1)
          rmul!(x1, 1 / Norm)
          rmul!(Ψ, Norm * exp(dt * (E₀ - E_shift)))

          if si < L
               info_forward[si] = TDVPInfo{1}(dt, info_Lanczos, BondInfo(x1, :R))

               @timeit TimerStep "svd" Ψ[si], S::MPSTensor, info_svd = leftorth(x1; trunc=trunc)
               # note svd may change the norm of S
               normalize!(S)
               
               # backward evolution
               @timeit TimerStep "pushEnv" canonicalize!(Env, si + 1, si)
               PH = CompositeProjectiveHamiltonian(Env.El[si + 1], Env.Er[si], (), E₀)
               @timeit TimerStep "TDVPUpdate0" S, info_Lanczos = LanczosExp(action, S, -dt, PH, TimerStep; K=K, tol=tol, verbose=false)
               finalize(PH)
               Norm = norm(S)
               rmul!(S, 1 / Norm)
               rmul!(Ψ, Norm * exp(-dt * (E₀ - E_shift)))

               # next Al
               Al = S * Ψ[si+1]
               # remember to change Center of Ψ manually
               Center(Ψ)[:] = [si + 1, si + 1]

               info_backward[si] = TDVPInfo{0}(-dt, info_Lanczos, info_svd)

               # update E₀, note Norm ~ exp(-dt * (⟨H⟩ - E₀))
               # E₀ -= log(Norm) / dt
          else
               Ψ[si] = x1
               info_forward[si] = TDVPInfo{1}(dt, info_Lanczos, BondInfo(x1, :R))
          end

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
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

     # GC manually
     GCsweep && manualGC(TimerSweep)

     return (forward=info_forward, backward=info_backward, cbe=info_cbe), TimerSweep

end

function TDVPSweep1!(Env::SparseEnvironment{L,3,T}, dt::Number, direction::SweepR2L; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     # right to left sweep
     K = get(kwargs, :K, 32)
     tol = get(kwargs, :tol, 1e-8)
     trunc = get(kwargs, :trunc, notrunc())
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     CBEAlg::CBEAlgorithm{SweepR2L} = get(kwargs, :CBEAlg, NoCBE())
     E_shift::Float64 = get(kwargs, :E_shift, 0.0)

     TimerSweep = TimerOutput()
     info_forward = Vector{TDVPInfo{1}}(undef, L)
     info_backward = Vector{TDVPInfo{0}}(undef, L - 1)
     info_cbe = Vector{Union{Nothing, CBEInfo}}(nothing, L - 1)

     Ψ = Env[3]
     canonicalize!(Ψ, L)
     canonicalize!(Env, L)
     # shift energy
     E₀::Float64 = scalar!(Env; normalize=true) |> real
     Ar::MPSTensor = Ψ[L]
     @timeit TimerSweep "TDVPSweep1<<" for si = reverse(1:L)
          TimerStep = TimerOutput()

          # CBE
          if !isa(CBEAlg, NoCBE) && si > 1
               canonicalize!(Env, si - 1, si)
               @timeit TimerStep "CBE" Ψ[si-1], Ar, info_cbe[si - 1], TO_CBE = CBE(Ψ[si-1], Ar, Env.El[si-1], Env.Er[si], Env[2][si-1], Env[2][si], CBEAlg)
               merge!(TimerStep, TO_CBE; tree_point=["CBE"])
          end

          @timeit TimerStep "pushEnv" canonicalize!(Env, si)

          PH = CompositeProjectiveHamiltonian(Env.El[si], Env.Er[si], (Env[2][si],), E₀)
          @timeit TimerStep "TDVPUpdate1" x1, info_Lanczos = LanczosExp(action, Ar, dt, PH, TimerStep; K=K, tol=tol, verbose=false)
          finalize(PH)
          Norm = norm(x1)
          rmul!(x1, 1 / Norm)
          rmul!(Ψ, Norm * exp(dt * (E₀ - E_shift)))

          if si > 1
               info_forward[si] = TDVPInfo{1}(dt, info_Lanczos, BondInfo(x1, :L))

               @timeit TimerStep "svd" S::MPSTensor, Ψ[si], info_svd = rightorth(x1; trunc=trunc)
               # note svd may change the norm of S
               normalize!(S)
               
               # backward evolution
               @timeit TimerStep "pushEnv" canonicalize!(Env, si, si - 1)

               PH = CompositeProjectiveHamiltonian(Env.El[si], Env.Er[si - 1], (), E₀)
               @timeit TimerStep "TDVPUpdate0" S, info_Lanczos = LanczosExp(action, S, -dt, PH, TimerStep; K=K, tol=tol, verbose=false)
               finalize(PH)
               Norm = norm(S)
               rmul!(S, 1 / Norm)
               rmul!(Ψ, Norm * exp(-dt * (E₀-E_shift)))

               # next Ar
               Ar = Ψ[si-1] * S
               # remember to change Center of Ψ manually
               Center(Ψ)[:] = [si - 1, si - 1]

               info_backward[si-1] = TDVPInfo{0}(-dt, info_Lanczos, info_svd)

               # update E₀, note Norm ~ exp(-dt * (⟨H⟩ - E₀))
               # E₀ -= log(Norm) / dt

          else
               Ψ[si] = x1
               info_forward[si] = TDVPInfo{1}(dt, info_Lanczos, BondInfo(x1, :L))
          end

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
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

     # GC manually
     GCsweep && manualGC(TimerSweep)

     return (forward=info_forward, backward=info_backward, cbe=info_cbe), TimerSweep

end

