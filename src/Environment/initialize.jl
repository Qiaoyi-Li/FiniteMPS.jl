function _initializeEnv!(obj::AbstractEnvironment{L}; kwargs...) where L
     # initialize boundary El and Er
     obj.El[1] = get(kwargs, :El, _defaultEl(obj))
     obj.Er[end] = get(kwargs, :Er, _defaultEr(obj))
     obj.Center[:] = [1, L]
     return obj
end

# ⟨Ψ₁, Ψ₂⟩
function _defaultEl(obj::SimpleEnvironment{L, 2, T}) where {L, T<:Tuple{AdjointMPS, DenseMPS}}
     return isometry(domain(obj[1][1])[1], codomain(obj[2][1])[1])
end
function _defaultEr(obj::SimpleEnvironment{L, 2, T}) where {L, T<:Tuple{AdjointMPS, DenseMPS}}
     return isometry(domain(obj[2][end])[end], codomain(obj[1][end])[end])
end

# ================ ⟨Ψ₁, H, Ψ₂⟩, sparse case ==================
function _defaultEl(obj::SparseEnvironment{L, 3, T}) where {L, T<:Tuple{AdjointMPS, SparseMPO, DenseMPS}}
     n = size(obj[2][1], 1)
     El = SparseLeftTensor(undef, n)
     for i = 1:n
          idx = findfirst(!isnothing, view(obj[2][1], i,:))
          El[i] = _simpleEl(obj[1][1], obj[2][1][i, idx], obj[3][1])
     end
     return El
end

function _simpleEl(A::AdjointMPSTensor, ::T, B::MPSTensor) where T <:Union{IdentityOperator, LocalOperator{1,R₂} where R₂}
     # no horizontal bond
     return isometry(domain(A)[1], codomain(B)[1])
end

function _simpleEl(A::AdjointMPSTensor, H::LocalOperator{2, R₂}, B::MPSTensor) where R₂
     return isometry(domain(A)[1], codomain(H)[1] ⊗ codomain(B)[1])
end

function _defaultEr(obj::SparseEnvironment{L, 3, T}) where {L, T<:Tuple{AdjointMPS, SparseMPO, DenseMPS}}
     n = size(obj[2][end], 2)
     Er = SparseRightTensor(undef, n)
     for i = 1:n
          idx = findfirst(!isnothing, view(obj[2][end], :,i))
          Er[i] = _simpleEr(obj[1][end], obj[2][end][idx, i], obj[3][end])
     end
     return Er
end

function _simpleEr(A::AdjointMPSTensor, ::T, B::MPSTensor) where T <:Union{IdentityOperator, LocalOperator{R₁,1} where R₁}
     # no horizontal bond
     return isometry(domain(B)[end], codomain(A)[end])
end
# ========================================================

