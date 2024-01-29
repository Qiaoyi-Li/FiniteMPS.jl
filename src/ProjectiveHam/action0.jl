"""
     action0(obj::SparseProjectiveHamiltonian{0}, x::MPSTensor{2}; kwargs...) -> ::MPSTensor

Action of 0-site projective Hamiltonian on the rank-2 bond local tensors.
"""
function action0(obj::SparseProjectiveHamiltonian{0}, x::MPSTensor{2}; kwargs...)

     Timer_action0 = get_timer("action0")
     @timeit Timer_action0 "action0" begin
          if get_num_workers() > 1
               f = (x, y) -> (add!(x[1], y[1]), merge!(x[2], y[2]))
               Hx, Timer_acc = @distributed (f) for (i,) in obj.validIdx
                    _action0(x, obj.El[i], obj.Er[i], true; kwargs...)
               end

          else

               Hx = nothing
               Timer_acc = TimerOutput()
               Lock = Threads.ReentrantLock()
               idx = Threads.Atomic{Int64}(1)
               Threads.@sync for t in 1:Threads.nthreads()
                    Threads.@spawn while true
                         idx_t = Threads.atomic_add!(idx, 1)
                         idx_t > length(obj.validIdx) && break

                         (i,) = obj.validIdx[idx_t]
                         tmp, to = _action0(x, obj.El[i], obj.Er[i], true; kwargs...)

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

          end
     end

     merge!(Timer_action0, Timer_acc; tree_point=["action0"])

     # x -> (H - E₀)x
     !iszero(obj.E₀) && axpy!(-obj.E₀, x.A, Hx)
     return MPSTensor(Hx)

end
function action0(obj::SparseProjectiveHamiltonian{0}, x::AbstractTensorMap; kwargs...)
     @assert rank(x) == 2
     return action0(obj, MPSTensor(x); kwargs...)
end

# ====================== wrap _action0 to test performance ==================
function _action0(x::MPSTensor, El::LocalLeftTensor{N₁}, Er::LocalRightTensor{N₂},
     timeit::Bool; kwargs...) where {N₁,N₂}

     !timeit && return _action0(x, El, Er; kwargs...), TimerOutput()

     LocalTimer = TimerOutput()
     name = "_action0_$(N₁)_$(N₂)"
     @timeit LocalTimer name Hx = _action0(x, El, Er; kwargs...)

     return Hx, LocalTimer
end

#  ========================= rank-2 MPS tensor ========================
#  --c(D)----d(D)--
# |                |
# |------b(χ)------|
# |                |
#  --a(D)    e(D)--
function _action0(x::MPSTensor{2}, El::LocalLeftTensor{2}, Er::LocalRightTensor{2}; kwargs...)
     return El.A * x.A * Er.A
end
function _action0(x::MPSTensor{2}, El::LocalLeftTensor{3}, Er::LocalRightTensor{3}; kwargs...)
     @tensor Hx[a; e] := (El.A[a b c] * x.A[c d]) * Er.A[d b e]
     return Hx
end
