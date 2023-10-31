# MPS is an element of Hilbert space, the algebra struct of Hilbert space should be implemented

# TODO, LinearCombination, +, -, /, inner, adjoint, \oplus
# function LinearCombination(A::AbstractMPS, cA::Number, B::AbstractMPS, cB::Number)
#      # return C = cA*A + cB*B
#      @assert A.L == B.L && A.Center == B.Center

#      NumType = reduce(promote_type, [A.NumType, B.NumType, typeof(cA), typeof(cB)])

#      c_correct = NumType[abs(cA)*exp(A.logNorm), abs(cB)*exp(B.logNorm)]
#      normC = max(abs.(c_correct)...)
#      normC != 0 ? c_correct /= normC : normC = 1

#      EmbA = Vector{AbstractTensorMap}(undef, A.L - 1)
#      EmbB = Vector{AbstractTensorMap}(undef, A.L - 1)
#      @floop WorkStealingEx() for si = 1:A.L-1
#           @inbounds EmbA[si], EmbB[si] = oplusEmbed(A[si], B[si], rank(A[si]))
#      end

#      @floop begin
#           WorkStealingEx()
#           C = Vector{AbstractTensorMap}(undef, A.L)
#           for si in 1:A.L
#                @inbounds(
#                     if si == 1
#                          @tensor A_p[-1 -2; -3] := A[si][-1 -2 1] * EmbA[si][1 -3]
#                          @tensor B_p[-1 -2; -3] := B[si][-1 -2 1] * EmbB[si][1 -3]
#                          C[si] = (cA / abs(cA)) * c_correct[1] * A_p + (cB / abs(cB)) * c_correct[2] * B_p
#                     elseif si == A.L
#                          @tensor A_p[-1 -2; -3] := A[si][1 -2 -3] * EmbA[si-1]'[-1 1]
#                          @tensor B_p[-1 -2; -3] := B[si][1 -2 -3] * EmbB[si-1]'[-1 1]
#                          C[si] = A_p + B_p
#                     else
#                          @tensor A_p[-1 -2; -3] := EmbA[si-1]'[-1 1] * A[si][1 -2 2] * EmbA[si][2 -3]
#                          @tensor B_p[-1 -2; -3] := EmbB[si-1]'[-1 1] * B[si][1 -2 2] * EmbB[si][2 -3]
#                          C[si] = A_p + B_p
#                     end
#                )
#            end
#      end

#      C = MPS(C)
#      # canonicalize
#      canonicalize!(C, 1)
#      C.logNorm += log(normC)
#      return C
# end

# +(A::AbstractMPS, B::AbstractMPS) = LinearCombination(A, 1, B, 1) 
# -(A::AbstractMPS, B::AbstractMPS) = LinearCombination(A, 1, B, -1) 

# # scalar multiple
# function *(A::AbstractMPS, a::Number)
#      C = deepcopy(A)
#      if typeof(a) <: Real 
#           C.logNorm += log(abs2(a))/2
#           C[C.Center[1] + 1] *= sign(a)
#      else
#           C.logNorm += log(abs2(a))/2
#           C[C.Center[1] + 1] *= cis(angle(a))
#           # maybe A is a real MPS and a is complex
#           if C.NumType <: Real 
#                C.NumType = promote_type(C.NumType, typeof(a))
#                for i in 1:C.L
#                     C[i] *= one(C.NumType)
#                end
#           end
#      end
#      return C
# end
# *(a::Number, A::AbstractMPS) = *(A, a)

# function /(A::AbstractMPS, a::Number) 
#      @assert a != 0
#      return *(A, 1/a) 
# end


