"""
     struct TDVPIntegrator{N} 
          dt::NTuple{N, Float64}
          direction::NTuple{N, SweepDirection}
     end

Type of TDVP integrators using composition methods. Note `dt` is relative, thus `sum(dt) == 1`.

# Constructors
     TDVPIntegrator(dt::NTuple{N, Rational}, direction::NTuple{N, SweepDirection})
Standard constructor with checking `dt` and `direction`.
     
     TDVPIntegrator(dt::Number...)
Assume direction is repeated as L-R-L-R-..., thus `length(dt)` must be even.
"""
struct TDVPIntegrator{N}
     dt::NTuple{N,Float64}
     direction::NTuple{N,SweepDirection}
     function TDVPIntegrator(dt::NTuple{N,<:Real}, direction::NTuple{N,SweepDirection}) where {N}
          # check accumulative time step is 1
          @assert sum(dt) == 1
          # same direction twice in a row is not allowed
          @assert all(direction[1:end-1] .!= direction[2:end])
          return new{N}(convert.(Float64, dt), direction)
     end
     function TDVPIntegrator(dt::Number...)
          # assume L-R-L-R-...
          @assert iseven(length(dt))
          direction = SweepDirection[]
          for i in 1:2:length(dt)
               push!(direction, SweepL2R())
               push!(direction, SweepR2L())
          end
          return TDVPIntegrator(convert.(Float64, Tuple(dt)), Tuple(direction))
     end
end

"""
     SymmetricIntegrator(p::Int64) -> TDVPIntegrator

Construct predefined symmetric integrators with `p`-th order. Only `p = 2, 3, 4` are supported.
"""
function SymmetricIntegrator(p::Int64)
     @assert p in (2, 3, 4)
     p == 2 && return TDVPIntegrator(1 // 2, 1 // 2)
     p == 3 && return TDVPIntegrator(1 // 8, 1 // 4, 1 // 8, 1 // 8, 1 // 4, 1 // 8) # arXiv:2208.10972
     if p == 4
          # Suzuki fractals
          # note this is a simple implementation, maybe there is a better one by using adjoint integrators 
          γ₁ = 1 / (4 - 4^(1 / 3))
          γ₂ = 1 - 4 * γ₁
          return TDVPIntegrator(γ₁ / 2, γ₁ / 2, γ₁ / 2, γ₁ / 2, γ₂ / 2, γ₂ / 2, γ₁ / 2, γ₁ / 2, γ₁ / 2, γ₁ / 2)
     end
end

for func in (:TDVPSweep1!, :TDVPSweep2!)
     @eval begin
          function $func(Env::SparseEnvironment{L,3,T},
               dt::Number,
               integrator::TDVPIntegrator{N} = SymmetricIntegrator(2);
               kwargs...) where {L,N,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}

               verbose::Int64 = get(kwargs, :verbose, 0)
               lsinfo = Vector{Any}(undef, N)
               lsTimer = Vector{TimerOutput}(undef, N)
               
               for i in 1:N
                    lsinfo[i], lsTimer[i] = $func(Env, dt * integrator.dt[i], integrator.direction[i]; kwargs...)

                    if verbose ≥ 1
                         str = isa(integrator.direction[i], SweepL2R) ? ">>" : "<<"
                         show(lsTimer[i]; title="TDVP sweep $(str) $(i)/$(N)")
                         let K = maximum(x -> x.Lanczos.numops, lsinfo[i].forward)
                              bondinfo_merge = merge(map(x -> x.Bond, lsinfo[i].forward))
                              println("\nForward: K = $(K), $(bondinfo_merge)")
                         end
                         let K = maximum(x -> x.Lanczos.numops, lsinfo[i].backward)
                              bondinfo_merge = merge(map(x -> x.Bond, lsinfo[i].backward))
                              println("Backward: K = $(K), $(bondinfo_merge)")
                         end
                         flush(stdout)
                    end
               end

               return lsinfo, lsTimer 
          end
     end
end