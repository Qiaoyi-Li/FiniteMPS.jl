"""
     action2(obj::SparseProjectiveHamiltonian{2}, x::CompositeMPSTensor{2, T}; kwargs...) -> ::CompositeMPSTensor{2, T}

Action of 2-site projective Hamiltonian on the 2-site local tensors, wrapped by `CompositeMPSTensor{2, T}` where `T<:NTuple{2,MPSTensor}`.

     action2(obj::IdentityProjectiveHamiltonian{2}, x::CompositeMPSTensor{2, T}; kwargs...) -> ::CompositeMPSTensor{2, T}
Special case for `IdentityProjectiveHamiltonian`.
"""
function action2(obj::SparseProjectiveHamiltonian{2}, x::CompositeMPSTensor{2,T}; kwargs...) where {T<:NTuple{2,MPSTensor}}

     Timer_action2 = get_timer("action2")
     @timeit Timer_action2 "action2" begin
          if get_num_workers() > 1  # multi-processing

               f = (x, y) -> (add!(x[1], y[1]), merge!(x[2], y[2]))
               Hx, Timer_acc = @distributed (f) for (i, j, k) in obj.validIdx
                    _action2(x, obj.El[i], obj.H[1][i, j], obj.H[2][j, k], obj.Er[k], true; kwargs...)
               end

          elseif get_num_threads_action() > 1

               # numthreads = Threads.nthreads()
               # # producer
               # taskref = Ref{Task}()
               # ch = Channel{Tuple{Int64,Int64,Int64}}(; taskref=taskref, spawn=true) do ch
               #      for idx in vcat(obj.validIdx, fill((0, 0, 0), numthreads))
               #           put!(ch, idx)
               #      end
               # end

               # Hx = nothing
               # Timer_acc = TimerOutput()

               # # consumers
               # Lock = Threads.SpinLock()
               # tasks = map(1:numthreads) do _
               #      task = Threads.@spawn while true
               #           (i, j, k) = take!(ch)
               #           i == 0 && break

               #           tmp, to = _action2(x, obj.El[i], obj.H[1][i, j], obj.H[2][j, k], obj.Er[k], true; kwargs...)

               #           lock(Lock)
               #           try
               #                Hx = axpy!(true, tmp, Hx)
               #                merge!(Timer_acc, to)
               #           catch 
               #                unlock(Lock)
               #                rethrow()
               #           end
               #           unlock(Lock)
               #      end
               #      errormonitor(task)
               # end

               # fetch.(tasks)
               # fetch(taskref[])

               Hx = nothing
               Timer_acc = TimerOutput()
               Lock = Threads.ReentrantLock()
               idx = Threads.Atomic{Int64}(1)
               Threads.@sync for _ in 1:Threads.nthreads()
                    Threads.@spawn while true
                         idx_t = Threads.atomic_add!(idx, 1)
                         idx_t > length(obj.validIdx) && break

                         (i, j, k) = obj.validIdx[idx_t]
                         tmp, to = _action2(x, obj.El[i], obj.H[1][i, j], obj.H[2][j, k], obj.Er[k], true; kwargs...)

                         lock(Lock)
                         try
                              Hx = axpy!(true, tmp, Hx)
                              merge!(Timer_acc, to)
                         catch
                              rethrow()
                         finally
                              unlock(Lock)
                         end
                    end
               end
          else
               Hx = nothing
               Timer_acc = TimerOutput()
               for (i, j, k) in obj.validIdx
                    tmp, to = _action2(x, obj.El[i], obj.H[1][i, j], obj.H[2][j, k], obj.Er[k], true; kwargs...)
                    Hx = axpy!(true, tmp, Hx)
                    merge!(Timer_acc, to)
               end

          end

     end

     merge!(Timer_action2, Timer_acc; tree_point=["action2"])

     # x -> (H - E₀)x
     !iszero(obj.E₀) && axpy!(-obj.E₀, x.A, Hx)

     return CompositeMPSTensor{2,T}(Hx)
end
function action2(obj::IdentityProjectiveHamiltonian{2}, x::CompositeMPSTensor{2,T}; kwargs...) where {T<:NTuple{2,MPSTensor}}
     Hx = _action2(x, obj.El,
          IdentityOperator(obj.si[1], 1),
          IdentityOperator(obj.si[2], 1),
          obj.Er; kwargs...)
     return CompositeMPSTensor{2,T}(Hx)
end
function action2(obj::AbstractProjectiveHamiltonian, x::AbstractVector{<:MPSTensor}; kwargs...)
     @assert length(x) == 2
     return action2(obj, CompositeMPSTensor(x[1], x[2]); kwargs...)
end
function action2(obj::AbstractProjectiveHamiltonian, xl::MPSTensor, xr::MPSTensor; kwargs...)
     return action2(obj, CompositeMPSTensor(xl, xr); kwargs...)
end

# ====================== wrap _action2 to test performance ==================
function _action2(x::CompositeMPSTensor, El::LocalLeftTensor{N₁}, Hl::LocalOperator{N₂,N₃}, Hr::LocalOperator{N₄,N₅}, Er::LocalRightTensor{N₆},
     timeit::Bool; kwargs...) where {N₁,N₂,N₃,N₄,N₅,N₆}

     !timeit && return _action2(x, El, Hl, Hr, Er; kwargs...), TimerOutput()

     LocalTimer = TimerOutput()
     name = "_action2_$(N₁)_$(N₂)$(N₃)_$(N₄)$(N₅)_$(N₆)"
     @timeit LocalTimer name Hx = _action2(x, El, Hl, Hr, Er; kwargs...)

     return Hx, LocalTimer
end

function _action2(x::CompositeMPSTensor, El::LocalLeftTensor{N₁}, Hl::IdentityOperator, Hr::LocalOperator{N₄,N₅}, Er::LocalRightTensor{N₆},
     timeit::Bool; kwargs...) where {N₁,N₄,N₅,N₆}

     !timeit && return _action2(x, El, Hl, Hr, Er; kwargs...), TimerOutput()

     LocalTimer = TimerOutput()
     name = "_action2_$(N₁)_0_$(N₄)$(N₅)_$(N₆)"
     @timeit LocalTimer name Hx = _action2(x, El, Hl, Hr, Er; kwargs...)

     return Hx, LocalTimer
end

function _action2(x::CompositeMPSTensor, El::LocalLeftTensor{N₁}, Hl::LocalOperator{N₂,N₃}, Hr::IdentityOperator, Er::LocalRightTensor{N₆},
     timeit::Bool; kwargs...) where {N₁,N₂,N₃,N₆}

     !timeit && return _action2(x, El, Hl, Hr, Er; kwargs...), TimerOutput()

     LocalTimer = TimerOutput()
     name = "_action2_$(N₁)_$(N₂)$(N₃)_0_$(N₆)"
     @timeit LocalTimer name Hx = _action2(x, El, Hl, Hr, Er; kwargs...)

     return Hx, LocalTimer
end

function _action2(x::CompositeMPSTensor, El::LocalLeftTensor{N₁}, Hl::IdentityOperator, Hr::IdentityOperator, Er::LocalRightTensor{N₆},
     timeit::Bool; kwargs...) where {N₁,N₆}

     !timeit && return _action2(x, El, Hl, Hr, Er; kwargs...), TimerOutput()

     LocalTimer = TimerOutput()
     name = "_action2_$(N₁)_0_0_$(N₆)"
     @timeit LocalTimer name Hx = _action2(x, El, Hl, Hr, Er; kwargs...)

     return Hx, LocalTimer
end
# -----------------------------------------------------------------------

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
     Hl::IdentityOperator, Hr::IdentityOperator,
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     coef = Hl.strength * Hr.strength
     @tensor Hx[a e; h i] := coef * (El.A[a b c] * x.A[c e h k]) * Er.A[k b i]
     return _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::IdentityOperator, Hr::LocalOperator{2,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     coef = Hl.strength * Hr.strength
     @tensor Hx[a e; g i] := coef * (El.A[a b c] * (x.A[c e h k] * Er.A[k i])) * Hr.A[b g h]
     return _permute2(Hx, x.A)
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
     Hl::LocalOperator{1,1}, Hr::LocalOperator{1,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     @tensor Hx[a d; g i] := (((El.A[a c] * x.A[c e h k]) * Hl.A[d e]) * Hr.A[g h]) * Er.A[k i]

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,2}, Hr::IdentityOperator,
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}
     coef = Hl.strength * Hr.strength
     @tensor Hx[a d; h i] := coef * ((El.A[a c] * x.A[c e h k]) * Hl.A[d e f]) * Er.A[k f i]
     return _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,2}, Hr::LocalOperator{1,1},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     # @tensor Hx[a d; g i] := (El.A[a c] * x.A[c e h k] * Hr.A[g h]) * (Hl.A[d e f] * Er.A[k f i]) # 2.9s/10(D = 8192)
     @tensor Hx[a d; g i] := ((El.A[a c] * x.A[c e h k] * Hr.A[g h]) * Hl.A[d e f]) * Er.A[k f i] # 2.1s/10(D=8192)

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,2}, Hr::LocalOperator{2,2},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     @tensor Hx[a d; g i] := ((El.A[a c] * x.A[c e h k] * Hl.A[d e f]) * Hr.A[f g h j]) * Er.A[k j i] 

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

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{2,2}, Hr::LocalOperator{2,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}
     @tensor Hx[a d; g i] := (El.A[a b c] * x.A[c e h k]) * (Hl.A[b d e f] * Hr.A[f g h]) * Er.A[k i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{2,2}, Hr::IdentityOperator,
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}
     @tensor Hx[a d; h i] := (El.A[a b c] * x.A[c e h k]) * Hl.A[b d e f] * Er.A[k f i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::IdentityOperator, Hr::LocalOperator{2,2},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}
     @tensor Hx[a e; g i] := El.A[a f c] * ((x.A[c e h k] * Er.A[k j i]) * Hr.A[f g h j]) 

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,2}, Hr::LocalOperator{2,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     @tensor Hx[a d; g i] := (El.A[a c] * x.A[c e h k]) * (Hl.A[d e f] * Hr.A[f g h]) * Er.A[k i] # 1.5s/10(D=8192)
     # @tensor Hx[a d; g i] := (((El.A[a c] * x.A[c e h k]) * Hl.A[d e f]) * Hr.A[f g h]) * Er.A[k i] # 1.8s/10(D=8192)
     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{2,1}, Hr::LocalOperator{1, 1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{3}}}

     @tensor Hx[a d; g i] := ((El.A[a b c] * (x.A[c e h k] * Hr.A[g h])) * Hl.A[b d e]) * Er.A[k i] 

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

# -----------------------------------------------------------------------
# ========================= 2 rank-4 MPO tensors ========================
#          l(d)     m(d) 
#           |        |
#   --c(D)-- -------- --k(D)--
#  |        |        |        |
#  |       e(d)     h(d)      |
#  |        |        |        |
#  |--b(χ)-- --f(χ)-- --j(χ)--|
#  |        |        |        |
#  |       d(d)     g(d)      |
#  |                          |
#   --a(D)              i(D)--
# TODO test performance and optimize contraction order, try prefusing El-Hl and Er-Hr
function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,1}, Hr::IdentityOperator,
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     @tensor Hx[a d l; h m i] := ((El.A[a c] * x.A[c e l h m k]) * Hl.A[d e]) * Er.A[k i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,2}, Hr::LocalOperator{2,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     @tensor Hx[a d l; g m i] := (El.A[a c] * x.A[c e l h m k]) * (Hl.A[d e f] * Hr.A[f g h]) * Er.A[k i]
     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,2}, Hr::LocalOperator{1,1},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     @tensor Hx[a d l; g m i] := ((El.A[a c] * x.A[c e l h m k] * Hr.A[g h]) * Hl.A[d e f]) * Er.A[k f i]
     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::IdentityOperator, Hr::LocalOperator{1,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     @tensor Hx[a e l; g m i] := ((El.A[a c] * x.A[c e l h m k]) * Hr.A[g h]) * Er.A[k i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::IdentityOperator, Hr::LocalOperator{1,2},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     @tensor Hx[a e l; g m i] := ((El.A[a c] * x.A[c e l h m k]) * Hr.A[g h j]) * Er.A[k j i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::IdentityOperator, Hr::IdentityOperator,
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     @tensor Hx[a e l; h m i] := (El.A[a c] * x.A[c e l h m k]) * Er.A[k i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{2,1}, Hr::IdentityOperator,
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     @tensor Hx[a d l; h m i] := El.A[a b c] * ((x.A[c e l h m k] * Er.A[k i]) * Hl.A[b d e])

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2}, Hl::LocalOperator{1,1}, Hr::LocalOperator{1,1}, Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}
     @tensor Hx[a d l; g m i] := (((El.A[a c] * x.A[c e l h m k]) * Hl.A[d e]) * Hr.A[g h]) * Er.A[k i]
     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{1,1}, Hr::LocalOperator{1,1},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     # @tensor Hx[a d l; g m i] := (((El.A[a b c] * x.A[c e l h m k]) * Hl.A[d e]) * Hr.A[g h]) * Er.A[k b i]
     @tensor Hx[a d l; g m i] := El.A[a b c] * ((x.A[c e l h m k] * Hl.A[d e]) * Hr.A[g h]) * Er.A[k b i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{1,1}, Hr::LocalOperator{2,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     @tensor Hx[a d l; g m i] := (((El.A[a b c] * x.A[c e l h m k]) * Hl.A[d e]) * Hr.A[b g h]) * Er.A[k i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{2,2}, Hr::LocalOperator{2,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}
     @tensor Hx[a d l; g m i] := (El.A[a b c] * x.A[c e l h m k]) * (Hl.A[b d e f] * Hr.A[f g h]) * Er.A[k i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{2,2}, Hr::LocalOperator{1,1},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     # @tensor Hx[a d l; g m i] := (((El.A[a b c] * x.A[c e l h m k]) * Hl.A[b d e f]) * Hr.A[g h]) * Er.A[k f i]
     @tensor Hx[a d l; g m i] := (El.A[a b c] * (x.A[c e l h m k] * Hr.A[g h])) * Hl.A[b d e f] * Er.A[k f i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::IdentityOperator, Hr::LocalOperator{2,2}, Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}
     @tensor Hx[a e l; g m i] := ((El.A[a b c] * x.A[c e l h m k]) * Hr.A[b g h j]) * Er.A[k j i]
     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::IdentityOperator, Hr::IdentityOperator, Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}
     @tensor Hx[a e l; h m i] := (El.A[a b c] * x.A[c e l h m k]) * Er.A[k b i]
     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,2}, Hr::IdentityOperator,
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}
     coef = Hl.strength * Hr.strength
     @tensor Hx[a d l; h m i] := coef * ((El.A[a c] * x.A[c e l h m k]) * Hl.A[d e f]) * Er.A[k f i]
     return _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::IdentityOperator, Hr::LocalOperator{2,1},
     Er::LocalRightTensor{2}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     coef = Hl.strength * Hr.strength
     @tensor Hx[a e l; g m i] := coef * (El.A[a b c] * (x.A[c e l h m k] * Er.A[k i])) * Hr.A[b g h]
     return _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{2},
     Hl::LocalOperator{1,2}, Hr::LocalOperator{2,2},
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}

     @tensor Hx[a d l; g m i] := ((El.A[a c] * x.A[c e l h m k] * Hl.A[d e f]) * Hr.A[f g h j]) * Er.A[k j i] 

     return Hl.strength * Hr.strength * _permute2(Hx, x.A)
end

function _action2(x::CompositeMPSTensor{2,T}, El::LocalLeftTensor{3},
     Hl::LocalOperator{2,2}, Hr::IdentityOperator,
     Er::LocalRightTensor{3}; kwargs...) where {T<:NTuple{2,MPSTensor{4}}}
     @tensor Hx[a d l; h m i] := (El.A[a b c] * x.A[c e l h m k]) * Hl.A[b d e f] * Er.A[k f i]

     return rmul!(_permute2(Hx, x.A), Hl.strength * Hr.strength)
end

# -----------------------------------------------------------------------

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
