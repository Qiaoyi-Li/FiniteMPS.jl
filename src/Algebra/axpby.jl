"""
     axpby!(α::Number, x::DenseMPS, β::Number, y::DenseMPS; kwargs...)

Compute `y = α*x + β*y` variationally via 2-site update, where `x` and `y` are dense MPS/MPO. Note 'x' cannot reference to the same MPS/MPO with `y`.  

# Kwargs
     trunc::TruncationScheme = truncbelow(MPSDefault.tol) & truncdim(MPSDefault.D)
     GCstep::Bool = false
     GCsweep::Bool = false
     maxiter::Int64 = 8
     disk::Bool = false
     tol::Float64 = 1e-8
     verbose::Int64 = 0
     lsnoise::AbstractVector{Float64} = Float64[]
"""
function axpby!(α::Number, x::DenseMPS{L}, β::Number, y::DenseMPS{L}; kwargs...) where {L}
     α == 0 && return rmul!(y, β)
     if β == 0 || coef(y) == 0
          # copy the data to y
          y.c = α * x.c
          y.Center[:] = x.Center[:]
          for i = 1:L
               y[i] = deepcopy(x[i])
          end
          return y
     end

     @assert !(x === y)

     trunc::TruncationScheme = get(kwargs, :trunc, truncbelow(MPSDefault.tol) & truncdim(MPSDefault.D))
     GCstep::Bool = get(kwargs, :GCstep, false)
     GCsweep::Bool = get(kwargs, :GCsweep, false)
     maxiter::Int64 = get(kwargs, :maxiter, 8)
     disk::Bool = get(kwargs, :disk, false)
     tol::Float64 = get(kwargs, :tol, 1e-8)
     verbose::Int64 = get(kwargs, :verbose, 0)
     lsnoise::Vector{Float64} = get(kwargs, :lsnoise, Float64[])

     Env_x = Environment(y', x; disk=disk)
     canonicalize!(Env_x, 1)
     Env_y = Environment(y', deepcopy(y); disk=disk)
     canonicalize!(Env_y, 1)

     # 2-site sweeps
     for iter = 1:maxiter
          TimerSweep = TimerOutput()
          direction::Symbol = :L2R
          convergence::Float64 = 0
          lsinfo = BondInfo[]
          @timeit TimerSweep "Sweep2" for si = vcat(1:L-1, reverse(1:L-1))

               TimerStep = TimerOutput()
               # 2-site local tensor before update
               convergence < tol && (A₀ = rmul!(CompositeMPSTensor(y[si], y[si+1]), coef(y)))


               @timeit TimerStep "pushEnv_x" canonicalize!(Env_x, si, si + 1)
               @timeit TimerStep "action2_x" Ax = action2(ProjHam(Env_x, si, si + 1), Env_x[2][si], Env_x[2][si+1]; kwargs...)
               @timeit TimerStep "pushEnv_y" canonicalize!(Env_y, si, si + 1)
               @timeit TimerStep "action2_y" Ay = action2(ProjHam(Env_y, si, si + 1), Env_y[2][si], Env_y[2][si+1]; kwargs...)

               A = axpby!(α * coef(x) / coef(y), Ax, β * coef(Env_y[2]) / coef(y), Ay)
               # normalize
               norm_A = norm(A)
               rmul!(A, 1 / norm_A)
               y.c *= norm_A

               # apply noise
               iter ≤ length(lsnoise) && noise!(A, lsnoise[iter])

               # svd
               if direction == :L2R
                    @timeit TimerStep "svd" y[si], y[si+1], svdinfo = leftorth(A; trunc=trunc, kwargs...)
                    Center(y)[:] = [si + 1, si + 1]
               else
                    @timeit TimerStep "svd" y[si], y[si+1], svdinfo = rightorth(A; trunc=trunc, kwargs...)
                    Center(y)[:] = [si, si]
               end
               push!(lsinfo, svdinfo)

               # check convergence
               if convergence < tol
                    @timeit TimerStep "convergence_check" begin
                         A = rmul!(CompositeMPSTensor(y[si], y[si+1]), coef(y))
                         convergence = max(convergence, norm(A - A₀)^2 / abs2(coef(y)))
                    end
               end
               # GC manually
               GCstep && manualGC(TimerStep)

               merge!(TimerSweep, TimerStep; tree_point=["Sweep2"])
               if verbose ≥ 2
                    ar = direction == :L2R ? "->" : "<-"
                    show(TimerStep; title="site $(si) $(ar) $(si+1)")
                    println("\niter $(iter), site $(si) $(ar) $(si+1), $(lsinfo[end]), max convergence = $(convergence)")
                    flush(stdout)
               end

               # change direction
               si == L - 1 && (direction = :R2L)

          end

          GCsweep && manualGC(TimerSweep)
          if verbose ≥ 1
               show(TimerSweep; title="axpby! iter $(iter)")
               println("\niter $(iter), $(merge(lsinfo)), max convergence = $(convergence) (tol = $(tol))")
               flush(stdout)
          end

          if iter > length(lsnoise) && convergence < tol
               break
          end
     end
     return y

end

axpy!(α::Number, x::DenseMPS, y::DenseMPS; kwargs...) = axpby!(α, x, 1, y; kwargs...)