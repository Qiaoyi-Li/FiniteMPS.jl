"""
     action2(obj::SparseProjectiveHamiltonian{2}, x::CompositeMPSTensor{2, T}; kwargs...) -> ::CompositeMPSTensor{2, T}

Action of 2-site projective Hamiltonian on the 2-site local tensors, wrapped by `CompositeMPSTensor{2, T}` where `T<:NTuple{2,MPSTensor}`.

# Kwargs
     distributed::Bool = false
If `ture`, use multi-processing, instead of multi-threading, for parallel computing.

     ntasks::Int64 = Threads.nthreads() - 1
Number ot tasks in multi-threading computation, details see `FoldsThreads.TaskPoolEx`.
"""
function action2(obj::SparseProjectiveHamiltonian{2}, x::CompositeMPSTensor{2,T}; kwargs...) where {T<:NTuple{2,MPSTensor}}

     if get(kwargs, :distributed, false)  # multi-processing

          tasks = map(obj.validIdx) do (i, j, k)
               let x = x, El = obj.El[i], Hl = obj.H[1][i, j], Hr = obj.H[2][j, k], Er = obj.Er[k]
                    @spawnat :any _action2(x, El, Hl, Hr, Er; kwargs...)
               end
          end
          Hx = mapreduce(fetch, (x, y) -> axpy!(true, x, y), tasks)

     else # multi-threading
          ntasks = get(kwargs, :ntasks, Threads.nthreads() - 1)

          @floop TaskPoolEx(; ntasks=ntasks, simd=true) for (i, j, k) in obj.validIdx
               tmp = _action2(x, obj.El[i], obj.H[1][i, j], obj.H[2][j, k], obj.Er[k]; kwargs...)
               @reduce() do (Hx = nothing; tmp)
                    Hx = axpy!(true, tmp, Hx)
               end
          end

     end
     return CompositeMPSTensor{2,T}(Hx)
end
function action2(obj::SparseProjectiveHamiltonian{2}, x::AbstractVector{<:MPSTensor}; kwargs...)
     @assert length(x) == 2
     return action2(obj, CompositeMPSTensor(x[1], x[2]); kwargs...)
end

# ========================= 2 rank-3 MPS tensors ========================
#   --c(D)-- -------- --k(D)--
#  |        |        |        |
#  |       e(d)     h(d)      |
#  |        |        |        |
#  |--b(χ)-- --f(χ)-- --j(χ)--|
#  |        |        |        |
#  |       d(d)     g(d)      |
#  |                          |
#   --a(D)              i(D)--
function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::IdentityOperator, Hr::IdentityOperator,
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     @tensor Hx[a e; h i] := (El.A[a c] * x.A[c e h k]) * Er.A[k i]

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{2,1}, Hr::IdentityOperator,
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     # @tensor Hx[a d; h i] := (El.A[a b c] * Hl.A[b d e]) * x.A[c e h k] * Er.A[k i] #2.6s/10(D=8192)
     # @tensor Hx[a d; h i] := (El.A[a b c] * Hl.A[b d e]) * (x.A[c e h k] * Er.A[k i]) #2.6s/10(D=8192)
     @tensor Hx[a d; h i] := ((El.A[a b c] * x.A[c e h k]) * Hl.A[b d e]) * Er.A[k i] #1.9s/10(D=8192)
     # @tensor Hx[a d; h i] := El.A[a b c] * (Hl.A[b d e]* (x.A[c e h k] * Er.A[k i])) #2.5s/10(D=8192)

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{1,1}, Hr::LocalOperator{2,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     # @tensor Hx[a d; g i] := (El.A[a b c] * Hr.A[b g h]) * (Hl.A[d e] * x.A[c e h k] * Er.A[k i]) # 3.1s/10(D = 8192)
     @tensor Hx[a d; g i] := (((El.A[a b c] * x.A[c e h k]) * Hl.A[d e]) * Hr.A[b g h]) * Er.A[k i] # 2.1s/10(D = 8192)
     # @tensor Hx[a d; g i] := ((El.A[a b c] * (x.A[c e h k] * Hl.A[d e])) * Hr.A[b g h]) * Er.A[k i] # 2.2s/10(D = 8192)

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{1,1}, Hr::LocalOperator{1,1},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     # @tensor Hx[a d; g i] := (El.A[a b c] * Hl.A[d e]) * x.A[c e h k] * (Hr.A[g h] * Er.A[k b i]) # 6.5s/10(D = 8192)
     # @tensor Hx[a d; g i] := ((El.A[a b c] * Hl.A[d e]) * x.A[c e h k] * Hr.A[g h]) * Er.A[k b i] # 4.3s/10(D = 8192)
     # @tensor Hx[a d; g i] := El.A[a b c] * (Hl.A[d e] * x.A[c e h k] * Hr.A[g h]) * Er.A[k b i] # 2.6s/10(D = 8192)
     @tensor Hx[a d; g i] := (((El.A[a b c] * x.A[c e h k]) * Hl.A[d e]) * Hr.A[g h]) * Er.A[k b i] # 2.6s/10(D = 8192)

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,1}, Hr::IdentityOperator,
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     # @tensor Hx[a d; h i] := (El.A[a c] * Hl.A[d e]) * x.A[c e h k] * Er.A[k i] # 2.5s/10(D=8192)
     # @tensor Hx[a d; h i] := El.A[a c] * (Hl.A[d e] * x.A[c e h k]) * Er.A[k i] # 1.8s/10(D=8192)
     @tensor Hx[a d; h i] := ((El.A[a c] * x.A[c e h k]) * Hl.A[d e]) * Er.A[k i] # 1.5s/10(D=8192)

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,2}, Hr::LocalOperator{1,1},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     # @tensor Hx[a d; g i] := (El.A[a c] * x.A[c e h k] * Hr.A[g h]) * (Hl.A[d e f] * Er.A[k f i]) # 2.9s/10(D = 8192)
     @tensor Hx[a d; g i] := ((El.A[a c] * x.A[c e h k] * Hr.A[g h]) * Hl.A[d e f]) * Er.A[k f i] # 2.1s/10(D=8192)

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::IdentityOperator, Hr::LocalOperator{1,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     # @tensor Hx[a e; g i] := (El.A[a c] * x.A[c e h k]) * (Hr.A[g h] * Er.A[k i]) # 2.6s/10(D=8192)
     @tensor Hx[a e; g i] := ((El.A[a c] * x.A[c e h k]) * Hr.A[g h]) * Er.A[k i] # 1.6s/10(D=8192)
     # @tensor Hx[a e; g i] := El.A[a c] * (Hr.A[g h] * (x.A[c e h k] * Er.A[k i])) #1.7s/10(D=8192)

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::IdentityOperator, Hr::LocalOperator{1,2},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     # @tensor Hx[a e; g i] := (El.A[a c] * x.A[c e h k]) * (Hr.A[g h j] * Er.A[k j i]) # 2.6s/10(D=8192)
     @tensor Hx[a e; g i] := ((El.A[a c] * x.A[c e h k]) * Hr.A[g h j]) * Er.A[k j i] # 2s/10(D=8192)

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,2}, Hr::LocalOperator{2,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     @tensor Hx[a d; g i] := (El.A[a c] * x.A[c e h k]) * (Hl.A[d e f] * Hr.A[f g h]) * Er.A[k i] # 1.5s/10(D=8192)
     # @tensor Hx[a d; g i] := (((El.A[a c] * x.A[c e h k]) * Hl.A[d e f]) * Hr.A[f g h]) * Er.A[k i] # 1.8s/10(D=8192)
     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end
# ================================================================


function _permute2(A::AbstractTensorMap, B::AbstractTensorMap)
     # permute A s.t. A and B have the same Index2Tuple

     r1 = rank(B, 1)
     r2 = rank(B, 2)
     if rank(A, 1) != r1
          return permute(A, Tuple(1:r1), Tuple(r1 .+ (1:r2)))
     else
          return A
     end
end
