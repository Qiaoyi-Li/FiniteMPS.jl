"""
     TDVPSweep2!(Env::SparseEnvironment{L,3,T},
          dt::Number,
          direction::SweepDirection;
          kwargs...) -> info, TimerSweep

Apply left-to-right or right-to-left 2-site TDVP`[https://doi.org/10.1103/PhysRevB.94.165116]` sweep to perform time evolution for `DenseMPS` (both MPS and MPO) with step length `dt`. `Env` is the 3-layer environment `⟨Ψ|H|Ψ⟩`.
     
     TDVPSweep2!(Env::SparseEnvironment{L,3,T}, dt::Number; kwargs...)
Wrap `TDVPSweep2!` with a symmetric integrator, i.e., sweeping from left to right and then from right to left with the same step length `dt / 2`.

# Kwargs
     K::Int64 = 32
     tol::Float64 = 1e-8
The maximum Krylov dimension and tolerance in the Lanczos exponential method.

     trunc = trunctol(; atol=MPSDefault.tol) & truncrank(MPSDefault.D)
     GCstep::Bool = false
     GCsweep::Bool = false
     verbose::Int64 = 0
     E_shift::Float64 = 0.0
Apply `exp(dt(H - E_shift))` to avoid possible `Inf` in imaginary time evolution. This energy shift is different from `E₀` in projective Hamiltonian, the later will give back the shifted energy thus not altering the final result. Note this is a temporary approach, we intend to store `log(norm)` in MPS to avoid this divergence in the future.
"""
function TDVPSweep2!(Env::SparseEnvironment{L,3,T}, dt::Number, ::SweepL2R; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     # left to right sweep
     K = get(kwargs, :K, 32)
     tol = get(kwargs, :tol, 1e-8)
     trunc = get(kwargs, :trunc, trunctol(; atol=MPSDefault.tol) & truncrank(MPSDefault.D))
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     E_shift::Float64 = get(kwargs, :E_shift, 0.0)

     TimerSweep = TimerOutput()
     info_forward = Vector{TDVPInfo{2}}(undef, L - 1)
     info_backward = Vector{TDVPInfo{1}}(undef, L - 2)

     Ψ = Env[3]
     # shift energy
     E₀::Float64 = scalar!(Env; normalize=true) |> real

     canonicalize!(Ψ, 1, 2)
     Al::MPSTensor = Ψ[1]
     @timeit TimerSweep "TDVPSweep2>>" for si in 1:L-1
          TimerStep = TimerOutput()
          @timeit TimerStep "pushEnv" canonicalize!(Env, si, si + 1)
          Ar = Ψ[si+1]

          PH = CompositeProjectiveHamiltonian(Env.El[si], Env.Er[si+1], (Env[2][si], Env[2][si+1]), E₀)
          @timeit TimerStep "TDVPUpdate2" x2, info_Lanczos = LanczosExp(action, CompositeMPSTensor(Al, Ar), dt, PH, TimerStep, ["TDVPUpdate2"]; K=K, tol=tol, verbose=false)
          finalize(PH)
          Norm = norm(x2)
          rmul!(x2, 1 / Norm)
          @timeit TimerStep "svd" Ψ[si], Al, info_svd = leftorth(x2; trunc=trunc)
          # note svd may change the norm of Al
          normalize!(Al)
          # remember to change Center of Ψ manually
          Center(Ψ)[:] = [si + 1, si + 1]
          rmul!(Ψ, Norm * exp(dt * (E₀ - E_shift)))
          info_forward[si] = TDVPInfo{2}(dt, info_Lanczos, info_svd)

          # backward evolution
          if si < L - 1
               @timeit TimerStep "pushEnv" canonicalize!(Env, si + 1, si + 1)
               PH = CompositeProjectiveHamiltonian(Env.El[si+1], Env.Er[si+1], (Env[2][si+1],), E₀)
               @timeit TimerStep "TDVPUpdate1" Al, info_Lanczos = LanczosExp(action, Al, -dt, PH, TimerStep, ["TDVPUpdate1"]; K=K, tol=tol, verbose=false)
               finalize(PH)
               Norm = norm(Al)
               rmul!(Al, 1 / Norm)
               rmul!(Ψ, Norm * exp(-dt * (E₀ - E_shift)))
               info_backward[si] = TDVPInfo{1}(-dt, info_Lanczos, BondInfo(Al, :R))

               # update E₀, note Norm ~ exp(-dt * (⟨H⟩ - E₀))
               # E₀ -= log(Norm) / dt
          end

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
          merge!(TimerSweep, TimerStep; tree_point=["TDVPSweep2>>"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si)->$(si+1)")
               let K = info_forward[si].Lanczos.numops, info_svd = info_forward[si].Bond
                    println("\nForward: K = $(K), $(info_svd)")
               end
               if si < L - 1
                    let K = info_backward[si].Lanczos.numops, info_svd = info_backward[si].Bond
                         println("Backward: K = $(K), $(info_svd)")
                    end
               end
               flush(stdout)
          end

     end
     Ψ[L] = Al

     # GC manually
     GCsweep && manualGC(TimerSweep)

     return (forward=info_forward, backward=info_backward), TimerSweep

end

function TDVPSweep2!(Env::SparseEnvironment{L,3,T}, dt::Number, ::SweepR2L; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     # right to left sweep
     K = get(kwargs, :K, 32)
     tol = get(kwargs, :tol, 1e-8)
     trunc = get(kwargs, :trunc, trunctol(; atol=MPSDefault.tol) & truncrank(MPSDefault.D))
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)
     verbose::Int64 = get(kwargs, :verbose, 0)
     E_shift::Float64 = get(kwargs, :E_shift, 0.0)

     TimerSweep = TimerOutput()
     info_forward = Vector{TDVPInfo{2}}(undef, L - 1)
     info_backward = Vector{TDVPInfo{1}}(undef, L - 2)

     Ψ = Env[3]
     # shift energy
     E₀::Float64 = scalar!(Env; normalize=true) |> real

     canonicalize!(Ψ, L - 1, L)
     Ar::MPSTensor = Ψ[L]
     @timeit TimerSweep "TDVPSweep2<<" for si = reverse(2:L)
          TimerStep = TimerOutput()
          @timeit TimerStep "pushEnv" canonicalize!(Env, si - 1, si)
          Al = Ψ[si-1]

          PH = CompositeProjectiveHamiltonian(Env.El[si-1], Env.Er[si], (Env[2][si-1], Env[2][si]), E₀)
          @timeit TimerStep "TDVPUpdate2" x2, info_Lanczos = LanczosExp(action, CompositeMPSTensor(Al, Ar), dt, PH, TimerStep, ["TDVPUpdate2"]; K=K, tol=tol, verbose=false)
          finalize(PH)
          Norm = norm(x2)
          rmul!(x2, 1 / Norm)
          @timeit TimerStep "svd" Ar, Ψ[si], info_svd = rightorth(x2; trunc=trunc)
          normalize!(Ar)
          # remember to change Center of Ψ manually
          Center(Ψ)[:] = [si - 1, si - 1]
          rmul!(Ψ, Norm * exp(dt * (E₀ - E_shift)))
          info_forward[si-1] = TDVPInfo{2}(dt, info_Lanczos, info_svd)

          # backward evolution
          if si > 2
               @timeit TimerStep "pushEnv" canonicalize!(Env, si - 1, si - 1)
               PH = CompositeProjectiveHamiltonian(Env.El[si-1], Env.Er[si-1], (Env[2][si-1],), E₀)
               @timeit TimerStep "TDVPUpdate1" Ar, info_Lanczos = LanczosExp(action, Ar, -dt, PH, TimerStep, ["TDVPUpdate1"]; K=K, tol=tol, verbose=false)
               finalize(PH)
               Norm = norm(Ar)
               rmul!(Ar, 1 / Norm)
               rmul!(Ψ, Norm * exp(-dt * (E₀ - E_shift)))
               info_backward[si-2] = TDVPInfo{1}(-dt, info_Lanczos, BondInfo(Ar, :L))

               # update E₀, note Norm ~ exp(-dt * (⟨H⟩ - E₀))
               # E₀ -= log(Norm) / dt
          end

          # GC manually
          GCstep && manualGC(TimerStep)

          # show
          merge!(TimerSweep, TimerStep; tree_point=["TDVPSweep2<<"])
          if verbose ≥ 2
               show(TimerStep; title="site $(si-1)<-$(si)")
               let K = info_forward[si-1].Lanczos.numops, info_svd = info_forward[si-1].Bond
                    println("\nForward: K = $(K), $(info_svd)")
               end
               if si > 2
                    let K = info_backward[si-2].Lanczos.numops, info_svd = info_backward[si-2].Bond
                         println("Backward: K = $(K), $(info_svd)")
                    end
               end
               flush(stdout)
          end
     end
     Ψ[1] = Ar

     # GC manually
     GCsweep && manualGC(TimerSweep)

     return (forward=info_forward, backward=info_backward), TimerSweep
end
