"""
     const SparseMPOTensor = Matrix{Union{Nothing, AbstractLocalOperator}} 
"""
const SparseMPOTensor = Matrix{Union{Nothing, AbstractLocalOperator}} 

function dim(M::SparseMPOTensor, idx::Int64)
     @assert idx == 1 || idx == 2
     sz = size(M)
     D = zeros(Int64, sz[idx])
     DD = zeros(Int64, sz[idx])
     if idx == 1
          for i = 1:sz[1]
               j = findfirst(x -> !isnothing(M[i, x]), 1:sz[2])
               isnothing(j) && continue
               D[i], DD[i] = _vdim(M[i, j], 1)
          end
     else
          for j = 1:sz[2]
               i = findfirst(x -> !isnothing(M[x, j]), 1:sz[1])
               isnothing(i) && continue
               D[j], DD[j] = _vdim(M[i, j], 2)
          end
     end
     return D, DD
end