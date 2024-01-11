"""
     connection!(obj::SparseEnvironment; 
          kwargs...) -> ::Matrix 

Return the connection `⟨∇⟨Hᵢ⟩, ∇⟨Hⱼ⟩⟩` where `H₀`, `H₁`, ⋯, `Hₙ` are the components of the total Hamiltonian, decomposed according to the boundary left environment. 

Note the state `obj[3]` must be right canonicalized. After this funcation, the canonical center will move to the right boundary. 

# kwargs
     moveback::Bool = false
Move the canonical center back to the left boundary if `true`.
"""
function connection!(obj::SparseEnvironment{L,3,T}; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     @assert Center(obj)[2] == 1
     Ψ = obj[3]
     @assert Center(Ψ)[2] == 1


     N = length(obj.El[1])
     lsEl = map(1:N) do i
          El = copy(obj.El[1])
          for idx in setdiff(1:N, i)
               El[idx] = nothing
          end
          El
     end

     conMat = zeros(typeof(coef(obj[3])), N, N)
     lsHx = Vector{MPSTensor}(undef, N)
     A::MPSTensor = Ψ[1]
     for si in 1:L
          for n in 1:N
               PH = SparseProjectiveHamiltonian(lsEl[n], obj.Er[si], (obj[2][si],), [si, si])
               lsHx[n] = action1(PH, A)
          end
          
          for i in 1:N, j in i:N
               conMat[i, j] += (inner(lsHx[i], lsHx[j]) - inner(lsHx[i], A) * inner(A, lsHx[j]))
          end

          # subtract the double counting due to gauge redundancy
          if si < L
               Ψ[si], S::MPSTensor = leftorth(A)
               for n in 1:N
                    lsEl[n] = _pushright(lsEl[n], Ψ[si]', obj[2][si], Ψ[si])
                    PH = SparseProjectiveHamiltonian(lsEl[n], obj.Er[si], (), [si+1, si])
                    lsHx[n] = action0(PH, S)
               end

               for i in 1:N, j in i:N
                    conMat[i, j] -= (inner(lsHx[i], lsHx[j]) - inner(lsHx[i], S) * inner(S, lsHx[j]))
               end

               A = S * Ψ[si+1]
          else
               # remember to update the last local tensor
               Ψ[si] = A
               Center(Ψ)[:] = [si, si]

               # note previous El and Er are no longer correct
               Center(obj)[:] = [1, L]
          end

     end

     # fill i > j
     for i in 1:N, j in 1: i-1
          conMat[i, j] = conMat[j, i]'
     end

     if get(kwargs, :moveback, false)
          canonicalize!(Ψ, 1)
          canonicalize!(obj, 1)
     end

     return conMat
end