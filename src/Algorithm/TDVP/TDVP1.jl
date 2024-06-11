"""
     TDVPSweep1!(Env::SparseEnvironment{L,3,T},
          dt::Number,
          direction::SweepDirection;
          kwargs...) -> info, TimerSweep

Apply 1-site TDVP`[https://doi.org/10.1103/PhysRevB.94.165116]` sweep to perform time evolution for `DenseMPS` (both MPS and MPO) with step length `dt`. `Env` is the 3-layer environment `⟨Ψ|H|Ψ⟩`.
     
     TDVPSweep1!(Env::SparseEnvironment{L,3,T}, dt::Number; kwargs...)
Wrap `TDVPSweep1!` with a symmetric integrator, i.e., sweeping from left to right and then from right to left with the same step length `dt / 2`.

# Kwargs
     krylovalg::KrylovKit.KrylovAlgorithm = TDVPDefaultLanczos
     trunc::TruncationType = notrunc()
     GCstep::Bool = false
     GCsweep::Bool = false
     verbose::Int64 = 0
     CBEAlg::CBEAlgorithm = NoCBE()
     E_shift::Float64 = 0.0
Apply `exp(dt(H - E_shift))` to avoid possible `Inf` in imaginary time evolution. This energy shift is different from `E₀` in projective Hamiltonian, the later will give back the shifted energy thus not altering the final result. Note this is a temporary approach, we intend to store `log(norm)` in MPS to avoid this divergence in the future.
"""
function TDVPSweep1!(Env::SparseEnvironment{L,3,T}, dt::Number, direction::SweepL2R; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     krylovalg = get(kwargs, :krylovalg, TDVPDefaultLanczos)
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
               @timeit TimerStep "CBE" Al, Ψ[si+1], info_cbe[si] = CBE(ProjHam(Env, si, si + 1; E₀=E₀), Al, Ψ[si+1], CBEAlg)
               merge!(TimerStep, get_timer("CBE"); tree_point=["CBE"])
          end

          @timeit TimerStep "pushEnv" canonicalize!(Env, si)
          @timeit TimerStep "TDVPUpdate1" x1, Norm, info_Lanczos = _TDVPUpdate1(ProjHam(Env, si; E₀=E₀), Al, dt, krylovalg; kwargs...)
          rmul!(Ψ, Norm * exp(dt * (E₀ - E_shift)))

          if si < L
               # TODO, test truncation after backward evolution
               @timeit TimerStep "svd" Ψ[si], S::MPSTensor, info_svd = leftorth(x1; trunc=trunc)
               # note svd may change the norm of S
               normalize!(S)
               info_forward[si] = TDVPInfo{1}(dt, info_Lanczos,  BondInfo(x1, :R))

               # backward evolution
               @timeit TimerStep "pushEnv" canonicalize!(Env, si + 1, si)
               @timeit TimerStep "TDVPUpdate0" S, Norm, info_Lanczos = _TDVPUpdate0(ProjHam(Env, si + 1, si; E₀=E₀), S, -dt, krylovalg; kwargs...)
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

     # GC manually
     GCsweep && manualGC(TimerSweep)

     return (forward=info_forward, backward=info_backward, cbe=info_cbe), TimerSweep

end

function TDVPSweep1!(Env::SparseEnvironment{L,3,T}, dt::Number, direction::SweepR2L; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     # right to left sweep
     krylovalg = get(kwargs, :krylovalg, TDVPDefaultLanczos)
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
               @timeit TimerStep "CBE" Ψ[si-1], Ar, info_cbe[si - 1] = CBE(ProjHam(Env, si - 1, si; E₀=E₀), Ψ[si-1], Ar, CBEAlg)
               merge!(TimerStep, get_timer("CBE"); tree_point=["CBE"])
          end

          @timeit TimerStep "pushEnv" canonicalize!(Env, si)
          @timeit TimerStep "TDVPUpdate1" x1, Norm, info_Lanczos = _TDVPUpdate1(ProjHam(Env, si; E₀=E₀), Ar, dt, krylovalg; kwargs...)
          rmul!(Ψ, Norm * exp(dt * (E₀ - E_shift)))

          if si > 1
               @timeit TimerStep "svd" S::MPSTensor, Ψ[si], info_svd = rightorth(x1; trunc=trunc)
               # note svd may change the norm of S
               normalize!(S)
               info_forward[si] = TDVPInfo{1}(dt, info_Lanczos, BondInfo(x1, :L))

               # backward evolution
               @timeit TimerStep "pushEnv" canonicalize!(Env, si, si - 1)
               @timeit TimerStep "TDVPUpdate0" S, Norm, info_Lanczos = _TDVPUpdate0(ProjHam(Env, si, si - 1; E₀=E₀), S, -dt, krylovalg; kwargs...)
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

     # GC manually
     GCsweep && manualGC(TimerSweep)

     return (forward=info_forward, backward=info_backward, cbe=info_cbe), TimerSweep

end

