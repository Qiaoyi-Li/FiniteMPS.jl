"""
     scalar!(obj::AbstractEnvironment; kwargs...) -> ::Number

Fully contract the total tensor network to get a scalar. 

Note this may change the `Center` of the environment `obj`.

# Kwargs
     normalize::Bool = false
If `true`, calculate `⟨Ψ₁|H|Ψ₂⟩/⟨Ψ₁|Ψ₂⟩` instead of `⟨Ψ₁|H|Ψ₂⟩` for example.

     split::Bool = false
Split the value into each contribution of each left boundary environment if `true`. Thus, return a vector instead of a scalar in this case. 
"""
function scalar!(obj::AbstractEnvironment; kwargs...) 

     if get(kwargs, :split, false)
          canonicalize!(obj, 1; free = false)
          return _scalar_split(obj; kwargs...)
     end

     # make sure there exist overlap between El and Er
     if obj.Center[1] ≤ obj.Center[2]
          si = max(Int64(ceil(sum(obj.Center) / 2)), 2)
          canonicalize!(obj, si, si - 1; free=false)
     end

     # contract El[si] and Er[si-1] to get the scalar
     return _scalar(obj, obj.Center[1]; kwargs...)

end

function _scalar(obj::SimpleEnvironment{L,2,T}, si::Int64; kwargs...) where {L,T<:Tuple{AdjointMPS,DenseMPS}}
     fac = coef(obj[1]) * coef(obj[2])
     if get(kwargs, :normalize, false) && (fac != 0)
          fac /= abs(fac)
     end
     return (obj.El[si] * obj.Er[si-1]) * fac
end

function _scalar(obj::SparseEnvironment{L,3,T}, si::Int64; kwargs...)  where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     @assert si > 1
     idx = filter(x -> !isnothing(obj.El[si][x]) && !isnothing(obj.Er[si-1][x]), eachindex(obj.El[si]))
     @floop GlobalThreadsExecutor for i in idx
          tmp = obj.El[si][i] * obj.Er[si-1][i]
          @reduce() do (acc = 0; tmp)
               acc += tmp
          end
     end

     fac = coef(obj[1]) * coef(obj[3])
     if get(kwargs, :normalize, false) && (fac != 0)
          fac /= abs(fac)
     end
     return acc * fac
end

function _scalar_split(obj::SparseEnvironment{L,3,T}; kwargs...) where {L, T <:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     @assert obj.Center[2] == 1

     Er = _pushleft(obj.Er[1], obj[1][1], obj[2][1], obj[3][1]; kwargs...)
     lsvalue = zeros(typeof(coef(obj[3])), length(obj.El[1]))

     idx = filter(x -> !isnothing(obj.El[1][x]) && !isnothing(Er[x]), eachindex(obj.El[1]))
     @floop GlobalThreadsExecutor for i in idx
          lsvalue[i] = obj.El[1][i] * Er[i]
     end

     fac = coef(obj[1]) * coef(obj[3])
     if get(kwargs, :normalize, false) && (fac != 0)
          fac /= abs(fac)
     end

     return rmul!(lsvalue, fac)
end

