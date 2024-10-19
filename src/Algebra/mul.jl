"""
     mul!(C::DenseMPS, A::SparseMPO, B::DenseMPS, α::Number, β::Number; kwargs...) 

Compute `C = α A*B + β C` variationally via 2-site update, where `A` is a sparse MPO, `B` and `C` are dense MPS/MPO. Note 'B' cannot reference to the same MPS/MPO with `C`.  

     mul!(C::DenseMPS, A::SparseMPO, B::DenseMPS; kwargs...)

Compute `C = A*B` by letting `α = 1` and `β = 0`.

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
function mul!(C::DenseMPS{L}, A::SparseMPO, B::DenseMPS{L}, α::Number, β::Number; kwargs...) where {L}
     @assert α != 0 || β != 0
     @assert !(B === C)

     trunc::TruncationScheme = get(kwargs, :trunc, truncbelow(MPSDefault.tol) & truncdim(MPSDefault.D))
     GCstep::Bool = get(kwargs, :GCstep, false)
     GCsweep::Bool = get(kwargs, :GCsweep, false)
     maxiter::Int64 = get(kwargs, :maxiter, 8)
     tol::Float64 = get(kwargs, :tol, 1e-8)
     verbose::Int64 = get(kwargs, :verbose, 0)
     lsnoise::Vector{Float64} = get(kwargs, :lsnoise, Float64[])

     if α != 0
          Env_mul = Environment(C', A, B; kwargs...)
          canonicalize!(Env_mul, 1)
     end
     if β != 0
          C₀ = deepcopy(C)
          Env_add = Environment(C', C₀; kwargs...)
          canonicalize!(Env_add, 1)
     end

     canonicalize!(C, 1)
     @assert coef(C) != 0
     # 2-site sweeps
     for iter = 1:maxiter
          TimerSweep = TimerOutput()
          direction::Symbol = :L2R
          convergence::Float64 = 0
          lsinfo = BondInfo[]
          @timeit TimerSweep "Sweep2" for si = vcat(1:L-1, reverse(1:L-1))

               TimerStep = TimerOutput()
               # 2-site local tensor before update
               convergence < tol && (x₀ = rmul!(CompositeMPSTensor(C[si], C[si+1]), coef(C)))

               if α != 0
                    @timeit TimerStep "pushEnv_mul" canonicalize!(Env_mul, si, si + 1)
                    @timeit TimerStep "action2_mul" ab = action2(ProjHam(Env_mul, si, si + 1), B[si], B[si+1]; kwargs...)
                    _α = α * coef(B) / coef(C)
               else
                    ab = nothing
                    _α = 0
               end
               if β != 0
                    @timeit TimerStep "pushEnv_add" canonicalize!(Env_add, si, si + 1)
                    @timeit TimerStep "action2_add" c = action2(ProjHam(Env_add, si, si + 1), C₀[si], C₀[si+1]; kwargs...)
                    _β = β * coef(C₀) / coef(C)
               else
                    c = nothing
                    _β = 0
               end

               x = axpby!(_α, ab, _β, c)
               # normalize
               norm_x = norm(x)
               rmul!(x, 1 / norm_x)
               C.c *= norm_x

               # apply noise
               iter ≤ length(lsnoise) && noise!(x, lsnoise[iter])

               # svd
               if direction == :L2R
                    @timeit TimerStep "svd" C[si], C[si+1], svdinfo = leftorth(x; trunc=trunc, kwargs...)
                    Center(C)[:] = [si + 1, si + 1]
               else
                    @timeit TimerStep "svd" C[si], C[si+1], svdinfo = rightorth(x; trunc=trunc, kwargs...)
                    Center(C)[:] = [si, si]
               end
               push!(lsinfo, svdinfo)

               # check convergence
               if convergence < tol
                    @timeit TimerStep "convergence_check" begin
                         x = rmul!(CompositeMPSTensor(C[si], C[si+1]), coef(C))
                         convergence = max(convergence, norm(x - x₀)^2 / abs2(coef(C)))
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
               show(TimerSweep; title="mul! iter $(iter)")
               println("\niter $(iter), $(merge(lsinfo)), max convergence = $(convergence) (tol = $(tol))")
               flush(stdout)
          end

          if iter > length(lsnoise) && convergence < tol
               break
          end

     end
     return C

end

function mul!(C::DenseMPS, A::SparseMPO, B::DenseMPS; kwargs...)
     # C = A*B
     return mul!(C, A, B, 1, 0; kwargs...)
end

function rmul!(A::DenseMPS, b::Number)
     # A -> b*A
     A.c *= b
     return A
end
