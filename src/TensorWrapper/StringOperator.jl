"""
     mutable struct StringOperator
          Ops::Vector{AbstractLocalOperator}
          strength::Number
     end

Concrete type for a closed or single-ended open string operator. The `Ops` field stores the local operators and `strength` is the overall strength.

# Constructor
     StringOperator(Ops::AbstractVector{<:AbstractLocalOperator}, strength::Number = 1.0) 
     StringOperator(Ops::AbstractLocalOperator..., strength::Number = 1.0)
     
# key methods 
     sort!(Ops::StringOperator)
Sort the operators by their site index in ascending order. Note the fermionic sign will be considered if necessary and will be absorbed into the `strength`.

     reduce!(Ops::StringOperator)
Reduce the string operator to a shorter one by numerically performing the composition of operators at the same site. Note this function can only be applied to a sorted string operator.
"""
mutable struct StringOperator
     Ops::Vector{AbstractLocalOperator}
     strength::Number
     function StringOperator(Ops::AbstractVector{<:AbstractLocalOperator}, strength::Number = 1.0)
          _validate_string(Ops)
          return new(AbstractLocalOperator[Ops...], strength * 1.0)
     end

     function StringOperator(A::AbstractLocalOperator, Args...)
          if !isempty(Args) && isa(Args[end], Number)
               return StringOperator(AbstractLocalOperator[A, Args[1:end-1]...], Args[end])
          else 
               return StringOperator(AbstractLocalOperator[A, Args...])
          end
     end
          
end

_string_auxspaces(O::LocalOperator{1,2}) = (nothing, domain(O.A, 2))
_string_auxspaces(O::LocalOperator{2,1}) = (codomain(O.A, 1), nothing)
_string_auxspaces(O::LocalOperator{2,2}) = (codomain(O.A, 1), domain(O.A, 2))
function _string_auxspaces(O::Union{LocalOperator{1,1},IdentityOperator})
     left, right = getLeftSpace(O), getRightSpace(O)
     left == right || throw(ArgumentError("unequal passthrough spaces at site $(O.si)"))
     return isunitspace(left) ? (nothing, nothing) : (left, right)
end
_string_auxspaces(O::AbstractLocalOperator) = throw(ArgumentError("unsupported string operator rank at site $(O.si)"))

function _validate_string(Ops; rightclosed::Bool = false)
     isempty(Ops) && throw(ArgumentError("a string operator must contain at least one operator"))
     left = nothing
     for O in Ops
          l, r = _string_auxspaces(O)
          if !isnothing(l) || !isnothing(r)
               left = l
               break
          end
     end
     current = left
     for O in Ops
          l, r = _string_auxspaces(O)
          if O isa Union{LocalOperator{1,1},IdentityOperator}
               isnothing(l) && continue
               current == l || throw(ArgumentError("auxiliary space mismatch at site $(O.si)"))
          else
               current == l || throw(ArgumentError("auxiliary bond mismatch at site $(O.si)"))
          end
          current = r
     end
     !isnothing(left) && !isnothing(current) && throw(ArgumentError("string operators open at both boundaries are not supported"))
     rightclosed && !isnothing(current) && throw(ArgumentError("string operator has an open right boundary"))
     return (; left, right = current)
end

length(obj::StringOperator) = length(obj.Ops)
for func in (:getindex, :lastindex, :setindex!, :iterate, :keys, :isassigned, :deleteat!)
     @eval Base.$func(obj::StringOperator, args...) = $func(obj.Ops, args...)
end
function show(io::IO, obj::StringOperator)
     L = length(obj)
     print(io, typeof(obj), "{$(L)}[")
     for i in 1:L
          show(io, obj.Ops[i])
          i < L && print(io, ", ")
     end 
     print(io, "]($(obj.strength))")
     return nothing
end 

function sort!(Ops::StringOperator)
     L = length(Ops)
     for j in 2:L 
          for i in j:-1:2 
               if Ops[i].si < Ops[i-1].si
                    Ops[i-1], Ops[i] = _swap(Ops[i-1], Ops[i])
                    if isfermionic(Ops[i-1]) && isfermionic(Ops[i]) 
                         Ops.strength *= -1
                    end
               else
                    break
               end
          end
     end
     _validate_string(Ops)
     return Ops
end

function reduce!(Ops::StringOperator)
     i = 1 
     while i < length(Ops)
          if Ops[i].si == Ops[i+1].si
               A, B = Ops[i], Ops[i+1]
               Ops[i] = A * B
               if A isa LocalOperator{1,1} && B isa LocalOperator{1,1}
                    V = isunitspace(getLeftSpace(A)) ? getLeftSpace(B) : getLeftSpace(A)
                    Ops[i].aspace = (V, V)
               end
               deleteat!(Ops, i+1)
          else
               i += 1
          end
     end
     _validate_string(Ops)
     return Ops
end

function _normalize_right_boundary!(S::StringOperator)
     boundary = _validate_string(S)
     isnothing(boundary.right) && return S
     return _normalize_open_boundary!(S)
end

function _normalize_open_boundary!(S::StringOperator)
     trunc = trunctol(; atol = 1e-15)
     firstactive = findfirst(S.Ops) do O
          !(O isa Union{LocalOperator{1,1},IdentityOperator} && isunitspace(getLeftSpace(O)))
     end
     carrier = nothing
     for i in length(S):-1:firstactive
          O = S[i]
          if O isa Union{LocalOperator{1,1},IdentityOperator}
               V = isnothing(carrier) ? unitspace(getPhysSpace(O)) : domain(carrier, 1)
               S[i] = _with_passthrough(O, V)
               continue
          end
          if isnothing(carrier)
               if i == firstactive
                    T = permute(O.A, ((3, 1), (2,)))
                    S[i] = LocalOperator(T, O.name, O.si, O.fermionic, O.strength)
                    break
               end
               p = O isa LocalOperator{1,2} ? ((3,), (1, 2)) : ((4, 1), (2, 3))
               U, D, Vh, _ = _raw_svd(O.A, p; trunc)
               carrier = U * D
               T = permute(Vh, ((1, 2), (3,)))
          else
               C = _contract_right_carrier(O, carrier)
               if i == firstactive
                    S[i] = LocalOperator(C, O.name, O.si, O.fermionic, O.strength)
                    break
               end
               p = O isa LocalOperator{1,2} ? ((1,), (2, 3, 4)) : ((1, 2), (3, 4, 5))
               U, D, Vh, _ = _raw_svd(C, p; trunc)
               carrier = U * D
               T = permute(Vh, ((1, 2), (3, 4)))
          end
          S[i] = LocalOperator(T, O.name, O.si, O.fermionic, O.strength)
     end
     _validate_string(S; rightclosed = true)
     return S
end

function _contract_right_carrier(O::LocalOperator{1,2}, carrier)
     @tensor allocator=ManualAllocator() C[q d; e b] := O.A[d e r] * carrier[q r b]
     return C
end
function _contract_right_carrier(O::LocalOperator{2,1}, carrier)
     @tensor allocator=ManualAllocator() C[q a d; e b] := O.A[a d e] * carrier[q b]
     return C
end
function _contract_right_carrier(O::LocalOperator{2,2}, carrier)
     @tensor allocator=ManualAllocator() C[q a d; e b] := O.A[a d e r] * carrier[q r b]
     return C
end

# swap two operators to deal with horizontal bond
_swap(A::LocalOperator{1, 1}, B::LocalOperator{1, 1}) = B, A
_with_passthrough(O::LocalOperator{1,1}, V) = LocalOperator(O.A, O.name, O.si, O.fermionic, O.strength, O.tag; aspace = (V, V))
_with_passthrough(O::IdentityOperator, V) = IdentityOperator(O.pspace, V, O.si, O.strength)
_swap(A::LocalOperator{1,1}, B::Union{LocalOperator{1,2},LocalOperator{2,1},LocalOperator{2,2}}) = B, _with_passthrough(A, getRightSpace(B))
_swap(A::Union{LocalOperator{1,2},LocalOperator{2,1},LocalOperator{2,2}}, B::LocalOperator{1,1}) = _with_passthrough(B, getLeftSpace(A)), A
function _swap(A::LocalOperator{1, 2}, B::LocalOperator{2, 1})
	return _swapOp(B), _swapOp(A)
end
function _swap(A::LocalOperator{1, 2}, B::LocalOperator{2, 2})
	#  |      |          |      |
	#  A--  --B--va -->  B--  --A--va
	#  |      |          |      |

	@tensor AB[d e; a b f] := A.A[a b c] * B.A[c d e f]
	# SVD
     TA, s, vd, _ = _raw_svd(AB; trunc = trunctol(; atol = 1e-15))
     TB = s * vd

	return LocalOperator(permute(TA, ((1,), (2, 3))), B.name, B.si, B.fermionic, B.strength), LocalOperator(permute(TB, ((1, 2), (3, 4))), A.name, A.si, A.fermionic, A.strength)
end
function _swap(A::LocalOperator{2, 2}, B::LocalOperator{2, 1})
	#     |     |         |     |
	# va--A-- --B --> va--B-- --A 
	#     |     |         |     |

	@tensor AB[a e f; b c] := A.A[a b c d] * B.A[d e f]
	# SVD, truncate zeros
     u, s, TB, _ = _raw_svd(AB; trunc = trunctol(; atol = 1e-15))
     TA = u * s

	return LocalOperator(permute(TA, ((1, 2), (3, 4))), B.name, B.si, B.fermionic, B.strength), LocalOperator(permute(TB, ((1, 2), (3,))), A.name, A.si, A.fermionic, A.strength)
end
function _swap(A::LocalOperator{2, 2}, B::LocalOperator{2, 2})
	#     |     |             |     |
	# va--A-- --B--vb --> va--B-- --A--vb 
	#     |     |             |     |

	@tensor AB[a e f; b c g] := A.A[a b c d] * B.A[d e f g]
	# SVD
     TA, s, vd, _ = _raw_svd(AB; trunc = trunctol(; atol = 1e-15))
	TB = s * vd

	return LocalOperator(permute(TA, ((1, 2), (3, 4))), B.name, B.si, B.fermionic, B.strength), LocalOperator(permute(TB, ((1, 2), (3, 4))), A.name, A.si, A.fermionic, A.strength)
end
function _swap(A::LocalOperator{2, 1}, B::LocalOperator{1, 2})
	#     |   |             |     |
	# va--A   B--vb --> va--B-- --A--vb 
	#     |   |             |     |

	@tensor AB[a e f; b c g] := A.A[a b c] * B.A[e f g]
	# SVD
     TA, s, vd, _ = _raw_svd(AB; trunc = trunctol(; atol = 1e-15))
     TB = s * vd

	return LocalOperator(permute(TA, ((1, 2), (3, 4))), B.name, B.si, B.fermionic, B.strength), LocalOperator(permute(TB, ((1, 2), (3, 4))), A.name, A.si, A.fermionic, A.strength)
end
