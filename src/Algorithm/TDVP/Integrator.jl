"""
     struct TDVPIntegrator{N} 
          dt::NTuple{N, Rational}
          direction::NTuple{N, Rational}
     end

Type of TDVP integrators using composition methods. Note `dt` is relative, thus `sum(dt) == 1`.

# Constructors
     TDVPIntegrator(dt::NTuple{N, Rational}, direction::NTuple{N, SweepDirection})
Standard constructor with checking `dt` and `direction`.
     
     TDVPIntegrator(dt::Number...)
Assume direction is repeated as L-R-L-R-..., thus `length(dt)` must be even.
"""
struct TDVPIntegrator{N} 
     dt::NTuple{N, Rational}
     direction::NTuple{N, SweepDirection}
     function TDVPIntegrator(dt::NTuple{N, Rational}, direction::NTuple{N, SweepDirection}) where N
          # check accumulative time step is 1
          @assert sum(dt) == 1
          # same direction twice in a row is not allowed
          @assert all(direction[1:end-1] .!= direction[2:end])
          return new{N}(dt, direction)
     end
     function TDVPIntegrator(dt::Number...) 
          # assume L-R-L-R-...
          @assert iseven(length(dt))
          direction = SweepDirection[]
          for i in 1:2:length(dt)
               push!(direction, SweepL2R())
               push!(direction, SweepR2L())
          end
          return TDVPIntegrator(convert.(Rational, dt), Tuple(direction))
     end
end

function SymmetricIntegrator(N::Int64)
     # TODO, implement some basic high-ordered symmetric integrators
     N == 2 && return TDVPIntegrator(1//2, 1//2) 

end