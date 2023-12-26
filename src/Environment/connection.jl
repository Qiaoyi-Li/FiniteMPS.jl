"""
     connection!(obj::SparseEnvironment,
          direction::SweepDirection; 
          kwargs...) -> ::Matrix 

Return the connection `⟨∇⟨Hᵢ⟩, ∇⟨Hⱼ⟩⟩` where `H₀`, `H₁`, ⋯, `Hₙ` are the components of the total Hamiltonian, decomposed according to the boundary left environment. 
"""
function connection!(obj::SparseEnvironment{L,3,T}, ::SweepL2R; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,DenseMPS}}
     @assert Center(obj)[2] == 1
     Ψ = obj[3]
     @assert Center(Ψ)[2] == 1

     disk::Bool = get(kwargs, :disk, false)


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
               conMat[i, j] += inner(lsHx[i], lsHx[j])
          end

          # subtract the double counting due to gauge redundancy
          if si < L
               Ψ[si], S::MPSTensor = leftorth(A)
               for n in 1:N
                    # TODO pushright, PH0
                    # lsEl[n] = 
               end
          end

     end



end