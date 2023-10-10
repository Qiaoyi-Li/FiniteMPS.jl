"""
     canonicalize!(obj::AbstractMPS, 
          siL::Int64
          [, siR::Int64 = siL]; kwargs...) -> obj::AbstractMPS

Canonicalize the MPS s.t. all sites ≤ `siL` are left-canonical, all sites ≥ `siR` are right-canonical.  

`kwargs` will be propagated to `leftorth` and `rightorth` to determine how to truncate the SVD spectra.
"""
function canonicalize!(obj::AbstractMPS{L}, siL::Int64, siR::Int64; kwargs...) where L
     @assert 1 ≤ siL ≤ siR ≤ L
     for si = Center(obj)[1]: siL - 1
          obj[si], R = leftorth(obj[si]; kwargs...)
          tmp = R*obj[si+1]
          obj.c *= norm(tmp)
          obj[si+1] = normalize!(tmp)
     end
     for si = Center(obj)[2]:-1:siR + 1
          Q, obj[si]= rightorth(obj[si]; kwargs...)
          tmp = obj[si-1]*Q
          obj.c *= norm(tmp)
          obj[si-1] = normalize!(tmp)
     end
     Center(obj)[:] = [siL, siR]
     return obj
end
function canonicalize!(obj::AbstractMPS, si::Vector{Int64}; kwargs...) 
     @assert length(si) == 2
     return canonicalize!(obj, si[1], si[2]; kwargs...)
end
canonicalize!(obj::AbstractMPS, si::Int64; kwargs...)  = canonicalize!(obj, si, si; kwargs...) 
