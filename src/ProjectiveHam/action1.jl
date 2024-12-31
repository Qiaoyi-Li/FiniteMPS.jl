"""
     action1(obj::SparseProjectiveHamiltonian{1}, x::MPSTensor; kwargs...) -> ::MPSTensor

Action of 1-site projective Hamiltonian on the 1-site local tensors.
"""
function action1(obj::SparseProjectiveHamiltonian{1}, x::MPSTensor; kwargs...)


     Timer_action1 = get_timer("action1")

     @timeit Timer_action1 "action1" begin
          if get_num_workers() > 1 # multi-processing
               f = (x, y) -> (add!(x[1], y[1]), merge!(x[2], y[2]))
               Hx, Timer_acc = @distributed (f) for (i, j) in obj.validIdx
                    _action1(x, obj.El[i], obj.H[1][i, j], obj.Er[j], true; kwargs...)
               end

          elseif get_num_threads_action() > 1

               Hx = nothing
               Timer_acc = TimerOutput()
               Lock = Threads.ReentrantLock()
               idx = Threads.Atomic{Int64}(1)
               Threads.@sync for _ in 1:Threads.nthreads()
                    Threads.@spawn while true
                         idx_t = Threads.atomic_add!(idx, 1)
                         idx_t > length(obj.validIdx) && break

                         (i, j) = obj.validIdx[idx_t]
                         tmp, to = _action1(x, obj.El[i], obj.H[1][i, j], obj.Er[j], true; kwargs...)

                         lock(Lock)
                         try
                              if isnothing(Hx)
                                   Hx = tmp  
                              else
                                   axpy!(true, tmp, Hx)
                              end
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
               for (i, j) in obj.validIdx
                    tmp, to = _action1(x, obj.El[i], obj.H[1][i, j], obj.Er[j], true; kwargs...)
                    Hx = axpy!(true, tmp, Hx)
                    merge!(Timer_acc, to)
               end
          end
     end

     merge!(Timer_action1, Timer_acc; tree_point=["action1"])

     # x -> (H - E₀)x
     !iszero(obj.E₀) && axpy!(-obj.E₀, x.A, Hx)

     return MPSTensor(Hx)
end
function action1(obj::SparseProjectiveHamiltonian{1}, x::AbstractTensorMap; kwargs...)
     return action1(obj, MPSTensor(x); kwargs...)
end

function action1(obj::PreFuseProjectiveHamiltonian{1}, x::MPSTensor; kwargs...)

     Timer_action1 = get_timer("action1")
     @timeit Timer_action1 "action1" begin
          if get_num_workers() > 1 # multi-processing

               f = (x, y) -> axpy!(true, x, y)
               Hx = @distributed (f) for j in 1:length(obj.El)
                    _action1(x, obj.El[j], obj.Er[j]; kwargs...)
               end

          else # multi-threading

               Hx = nothing
               Timer_acc = TimerOutput()
               Lock = Threads.ReentrantLock()
               idx = Threads.Atomic{Int64}(1)
               Threads.@sync for t in 1:Threads.nthreads()
                    Threads.@spawn while true
                         idx_t = Threads.atomic_add!(idx, 1)
                         idx_t > length(obj.El) && break

                         tmp, to = _action1(x, obj.El[idx_t], obj.Er[idx_t], true; kwargs...)

                         lock(Lock)
                         try
                              if isnothing(Hx)
                                   Hx = tmp  
                              else
                                   axpy!(true, tmp, Hx)
                              end
                              merge!(Timer_acc, to)
                         catch
                              rethrow()
                         finally
                              unlock(Lock)
                         end
                    end
               end

          end
     end

     merge!(Timer_action1, Timer_acc; tree_point=["action1"])

     # x -> (H - E₀)x
     !iszero(obj.E₀) && axpy!(-obj.E₀, x.A, Hx)

     return MPSTensor(Hx)
end

# ====================== wrap _action1 to test performance ==================
function _action1(x::MPSTensor, El::LocalLeftTensor{N₁}, H::LocalOperator{N₂,N₃}, Er::LocalRightTensor{N₄},
     timeit::Bool; kwargs...) where {N₁,N₂,N₃,N₄}

     !timeit && return _action1(x, El, H, Er; kwargs...), TimerOutput()

     LocalTimer = TimerOutput()
     name = "_action1_$(N₁)_$(N₂)$(N₃)_$(N₄)"
     @timeit LocalTimer name Hx = _action1(x, El, H, Er; kwargs...)

     return Hx, LocalTimer
end

function _action1(x::MPSTensor, El::LocalLeftTensor{N₁}, H::IdentityOperator, Er::LocalRightTensor{N₄},
     timeit::Bool; kwargs...) where {N₁,N₄}

     !timeit && return _action1(x, El, H, Er; kwargs...), TimerOutput()

     LocalTimer = TimerOutput()
     name = "_action1_$(N₁)_0_$(N₄)"
     @timeit LocalTimer name Hx = _action1(x, El, H, Er; kwargs...)

     return Hx, LocalTimer
end

function _action1(x::MPSTensor, El::LeftPreFuseTensor{N₁}, Er::LocalRightTensor{N₂},
     timeit::Bool; kwargs...) where {N₁,N₂}

     !timeit && return _action1(x, El, Er; kwargs...), TimerOutput()

     LocalTimer = TimerOutput()
     name = "_action1_$(N₁)_$(N₂)"
     @timeit LocalTimer name Hx = _action1(x, El, Er; kwargs...)

     return Hx, LocalTimer
end

# ========================= rank-3 MPS tensor ========================
#   --c(D)-- --h(D)--
#  |        |        |
#  |       e(d)      |
#  |        |        |
#  |--b(χ)-- --g(χ)--|
#  |        |        |    
#  |       d(d)      | 
#  |                 |
#   --a(D)     f(D)--
function _action1(x::MPSTensor{3}, El::LocalLeftTensor{2}, H::IdentityOperator, Er::LocalRightTensor{2}; kwargs...)
     # @tensor Hx[a e; f] := El.A[a c] * (x.A[c e h] * Er.A[h f])
     @tensor Hx[a e; f] := H.strength * El.A[a c] * x.A[c e h] * Er.A[h f]
     return Hx
end

function _action1(x::MPSTensor{3}, El::LocalLeftTensor{3}, H::IdentityOperator, Er::LocalRightTensor{3}; kwargs...)
     @tensor Hx[a e; f] := H.strength * (El.A[a b c] * x.A[c e h]) * Er.A[h b f]
     return Hx
end

function _action1(x::MPSTensor{3}, El::LocalLeftTensor{3}, H::LocalOperator{2,1}, Er::LocalRightTensor{2}; kwargs...)
     # @tensor Hx[a d; f] := (El.A[a b c] * (x.A[c e h] * Er.A[h f])) *  H.A[b d e]
     @tensor Hx[a d; f] := H.strength * El.A[a b c] * (x.A[c e h] * Er.A[h f]) * H.A[b d e]
     return Hx
end

function _action1(x::MPSTensor{3}, El::LocalLeftTensor{3}, H::LocalOperator{1,1}, Er::LocalRightTensor{3}; kwargs...)
     @tensor Hx[a d; f] := (El.A[a b c] * (H.A[d e] * x.A[c e h])) * Er.A[h b f]
     # @tensor Hx[a d; f] := El.A[a b c] * H.A[d e] * x.A[c e h] * Er.A[h b f]
     return H.strength * Hx
end

function _action1(x::MPSTensor{3}, El::LocalLeftTensor{2}, H::LocalOperator{1,1}, Er::LocalRightTensor{2}; kwargs...)
     # @tensor Hx[a d; f] := El.A[a c] * (H.A[d e] * (x.A[c e h] * Er.A[h f]))
     @tensor Hx[a d; f] := H.strength * El.A[a c] * x.A[c e h] * H.A[d e] * Er.A[h f]
     return Hx
end

function _action1(x::MPSTensor{3}, El::LocalLeftTensor{2}, H::LocalOperator{1,2}, Er::LocalRightTensor{3}; kwargs...)
     # @tensor Hx[a d; f] := El.A[a c] * x.A[c e h] * H.A[d e g] * Er.A[h g f]
     @tensor Hx[a d; f] := H.strength * El.A[a c] * x.A[c e h] * Er.A[h g f] * H.A[d e g]
     return Hx
end

function _action1(x::MPSTensor{3}, El::LocalLeftTensor{3}, H::LocalOperator{2,2}, Er::LocalRightTensor{3}; kwargs...)
     @tensor Hx[a d; f] := El.A[a b c] * x.A[c e h] * H.A[b d e g] * Er.A[h g f]
     return rmul!(Hx, H.strength)
end

# ========================= rank-4 MPO tensor ========================
#          i(d)
#           |
#   --c(D)-- --h(D)--
#  |        |        |
#  |       e(d)      |
#  |        |        |
#  |--b(χ)-- --g(χ)--|
#  |        |        |    
#  |       d(d)      | 
#  |                 |
#   --a(D)     f(D)--
function _action1(x::MPSTensor{4}, El::LocalLeftTensor{2}, H::IdentityOperator, Er::LocalRightTensor{2}; kwargs...)
     @tensor Hx[a e; i f] := El.A[a c] * x.A[c e i h] * Er.A[h f]
     return rmul!(Hx, H.strength)
end

function _action1(x::MPSTensor{4}, El::LocalLeftTensor{3}, H::IdentityOperator, Er::LocalRightTensor{3}; kwargs...)
     @tensor Hx[a e; i f] := (El.A[a b c] * x.A[c e i h]) * Er.A[h b f]
     return rmul!(Hx, H.strength)
end

function _action1(x::MPSTensor{4}, El::LocalLeftTensor{3}, H::LocalOperator{2,1}, Er::LocalRightTensor{2}; kwargs...)
     @tensor Hx[a d; i f] := El.A[a b c] * (x.A[c e i h] * Er.A[h f]) * H.A[b d e]
     return rmul!(Hx, H.strength)
end

function _action1(x::MPSTensor{4}, El::LocalLeftTensor{3}, H::LocalOperator{1,1}, Er::LocalRightTensor{3}; kwargs...)
     @tensor Hx[a d; i f] := El.A[a b c] * (H.A[d e] * x.A[c e i h]) * Er.A[h b f]
     return rmul!(Hx, H.strength)
end

function _action1(x::MPSTensor{4}, El::LocalLeftTensor{2}, H::LocalOperator{1,1}, Er::LocalRightTensor{2}; kwargs...)
     @tensor Hx[a d; i f] := El.A[a c] * (H.A[d e] * x.A[c e i h]) * Er.A[h f]
     return rmul!(Hx, H.strength)
end

function _action1(x::MPSTensor{4}, El::LocalLeftTensor{2}, H::LocalOperator{1,2}, Er::LocalRightTensor{3}; kwargs...)
     @tensor Hx[a d; i f] := El.A[a c] * x.A[c e i h] * H.A[d e g] * Er.A[h g f]
     return rmul!(Hx, H.strength)
end

function _action1(x::MPSTensor{4}, El::LocalLeftTensor{3}, H::LocalOperator{2,2}, Er::LocalRightTensor{3}; kwargs...)
     @tensor Hx[a d; i f] := El.A[a b c] * x.A[c e i h] * H.A[b d e g] * Er.A[h g f]
     return rmul!(Hx, H.strength)
end


# ================== prefuse version ===================
function _action1(x::MPSTensor{4}, El::LeftPreFuseTensor{4}, Er::LocalRightTensor{2}; kwargs...)
     @tensor Hx[a b; f h] := El.A[a b d e] * x.A[d e f g] * Er.A[g h]
     return Hx
end
function _action1(x::MPSTensor{4}, El::LeftPreFuseTensor{5}, Er::LocalRightTensor{3}; kwargs...)
     @tensor Hx[a b; f h] := El.A[a b c d e] * x.A[d e f g] * Er.A[g c h]
     return Hx
end