"""
     initialize!(obj::AbstractEnvironment; kwargs...)

Initialize the boundary environment tensors, i.e. `El[1]` and `Er[L]`. 

# Kwargs
     El::Union{SimpleLeftTensor, SparseLeftTensor}
     Er::Union{SimpleRightTensor, SparseRightTensor}

Directly give `El` or `Er`, otherwise, use `_defaultEl` or `_defaultEr` to generate one.

     free::Bool = false
If `true`, call `free!(obj)` to free the local environment tensors which are no longer required. Details see `free!`.
"""
function initialize!(obj::AbstractEnvironment{L}; kwargs...) where {L}
     obj.Center[:] = [1, L]
     obj.Er[end] = get(kwargs, :Er, _defaultEr(obj))
     obj.El[1] = get(kwargs, :El, _defaultEl(obj))
     if get(kwargs, :free, false)
          free!(obj)
     end
     return obj
end

# ⟨Ψ₁, Ψ₂⟩
function _defaultEl(obj::SimpleEnvironment{L,2,T}) where {L,T<:Tuple{AdjointMPS,DenseMPS}}
     return isometry(domain(obj[1][1])[1], codomain(obj[2][1])[1])
end
function _defaultEr(obj::SimpleEnvironment{L,2,T}) where {L,T<:Tuple{AdjointMPS,DenseMPS}}
     return isometry(domain(obj[2][end])[end], codomain(obj[1][end])[end])
end

# ================ ⟨Ψ₁, H, Ψ₂⟩, sparse case ==================
function _defaultEl(obj::SparseEnvironment{L,3,T}) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     try
          n = size(obj[2][1], 1)
          El = SparseLeftTensor(undef, n)
          for i = 1:n
               idx = findfirst(x -> !isnothing(x) && !isa(x, IdentityOperator), view(obj[2][1], i, :))
               @assert !isnothing(idx)
               El[i] = _simpleEl(obj[1][1], obj[2][1][i, idx], obj[3][1])
          end
          return El
     catch
          # cannot directly deduce the horizontal bond, contract from right to left to get it
          while obj.Center[2] > 1
               pushleft!(obj)
          end
          Er = _pushleft(obj.Er[1], obj[1][1], obj[2][1], obj[3][1])
          n = length(Er)
          El = SparseLeftTensor(undef, n)
          for i in 1:n
               @assert rank(Er[i]) in [2, 3]
               if rank(Er[i]) == 2
                    El[i] = _simpleEl(obj[1][1], IdentityOperator(1), obj[3][1])
               elseif rank(Er[i], 1) == 2
                    El[i] = _simpleEl(obj[1][1], codomain(Er[i])[2], obj[3][1])
               else
                    El[i] = _simpleEl(obj[1][1], domain(Er[i])[1]', obj[3][1])
               end 
          end
          return El
     end
end

function _simpleEl(A::AdjointMPSTensor, ::T, B::MPSTensor) where {T<:Union{IdentityOperator,LocalOperator{1,R₂} where R₂}}
     # no horizontal bond
     return isometry(domain(A)[1], codomain(B)[1])
end

function _simpleEl(A::AdjointMPSTensor, Vh::VectorSpace, B::MPSTensor)
     v = fuse(Vh, codomain(B)[1])
     iso = isometry(v, Vh ⊗ codomain(B)[1])
     for k in collect(keys(v.dims))
          k ∉ keys(domain(A)[1].dims) && delete!(v.dims, k)
     end
     emb = isometry(domain(A)[1], v)
     return emb * iso
end

function _simpleEl(A::AdjointMPSTensor, H::LocalOperator{2,R₂}, B::MPSTensor) where {R₂}
     return _simpleEl(A, codomain(H)[1], B)
end

function _defaultEr(obj::SparseEnvironment{L,3,T}) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     n = size(obj[2][end], 2)
     Er = SparseRightTensor(undef, n)
     for i = 1:n
          idx = findfirst(!isnothing, view(obj[2][end], :, i))
          Er[i] = _simpleEr(obj[1][end], obj[2][end][idx, i], obj[3][end])
     end
     return Er
end

function _simpleEr(A::AdjointMPSTensor, ::T, B::MPSTensor) where {T<:Union{IdentityOperator,LocalOperator{R₁,1} where R₁}}
     # no horizontal bond
     return isometry(domain(B)[end], codomain(A)[end])
end
# ========================================================

