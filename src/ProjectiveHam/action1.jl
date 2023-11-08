"""
     action1(obj::SparseProjectiveHamiltonian{1}, x::MPSTensor; kwargs...) -> ::MPSTensor

Action of 1-site projective Hamiltonian on the 1-site local tensors.
"""
function action1(obj::SparseProjectiveHamiltonian{1}, x::MPSTensor; kwargs...)

     if get_num_workers() > 1 # multi-processing

          f = (x, y) -> axpy!(true, x, y)
          Hx = @distributed (f) for (i, j) in obj.validIdx
               _action1(x, obj.El[i], obj.H[1][i, j], obj.Er[j]; kwargs...)
          end

     else # multi-threading

          @floop GlobalThreadsExecutor for (i, j) in obj.validIdx
               tmp = _action1(x, obj.El[i], obj.H[1][i, j], obj.Er[j]; kwargs...)
               @reduce() do (Hx = nothing; tmp)
                    Hx = axpy!(true, tmp, Hx)
               end
          end
     end

     # x -> (H - E₀)x
     !iszero(obj.E₀) && axpy!(-obj.E₀, x.A, Hx)

     return MPSTensor(Hx) 
end
function action1(obj::SparseProjectiveHamiltonian{1}, x::AbstractTensorMap; kwargs...)
     return action1(obj, MPSTensor(x); kwargs...)
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
     @tensor Hx[a e; f] := El.A[a c] * x.A[c e h] * Er.A[h f]
     return H.strength * Hx
end

function _action1(x::MPSTensor{3}, El::LocalLeftTensor{3}, H::LocalOperator{2,1}, Er::LocalRightTensor{2}; kwargs...)
     # @tensor Hx[a d; f] := (El.A[a b c] * (x.A[c e h] * Er.A[h f])) *  H.A[b d e]
     @tensor Hx[a d; f] := El.A[a b c] * x.A[c e h] * Er.A[h f] * H.A[b d e]
     return H.strength * Hx
end

function _action1(x::MPSTensor{3}, El::LocalLeftTensor{3}, H::LocalOperator{1,1}, Er::LocalRightTensor{3}; kwargs...)
     # @tensor Hx[a d; f] := (El.A[a b c] * (H.A[d e] * x.A[c e h])) * Er.A[h b f]
     @tensor Hx[a d; f] := El.A[a b c] * H.A[d e] * x.A[c e h] * Er.A[h b f]
     return H.strength * Hx
end

function _action1(x::MPSTensor{3}, El::LocalLeftTensor{2}, H::LocalOperator{1,1}, Er::LocalRightTensor{2}; kwargs...)
     # @tensor Hx[a d; f] := El.A[a c] * (H.A[d e] * (x.A[c e h] * Er.A[h f]))
     @tensor Hx[a d; f] := El.A[a c] * H.A[d e] * x.A[c e h] * Er.A[h f]
     return H.strength * Hx
end

function _action1(x::MPSTensor{3}, El::LocalLeftTensor{2}, H::LocalOperator{1,2}, Er::LocalRightTensor{3}; kwargs...)
     @tensor Hx[a d; f] := El.A[a c] * x.A[c e h] * H.A[d e g] * Er.A[h g f]
     return H.strength * Hx
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