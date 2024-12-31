"""
     mutable struct StringOperator
          Ops::Vector{AbstractLocalOperator}
          strength::Number
     end

Concrete type for an arbitrary string operator. The `Ops` field stores the local operators and `strength` is the overall strength.

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
          L = length(Ops)
          # check horizontal bonds 
          @assert rank(Ops[1], 1) == rank(Ops[end], 2) == 1
          @assert rank(Ops[1], 2) ∈ [1, 2]
          a_flag = rank(Ops[1], 2) == 2
          for i in 2:L 
               r1, r2 = rank(Ops[i],1), rank(Ops[i], 2)
               @assert r1 ∈ [1, 2] && r2 ∈ [1, 2] 
               r1 == r2 == 1 && continue
               if a_flag
                    # possible rank = (1, 1), (2, 1), (2, 2)
                    if (r1, r2) == (2, 1)
                         a_flag = false
                    elseif (r1, r2) != (2, 2)
                         error("rank mismatch at $(i-1)->$(i) bond!")
                    end
               else
                    # possible rank = (1, 1), (1, 2)
                    if (r1, r2) == (1, 2)
                         a_flag = true
                    else
                         error("rank mismatch at $(i-1)->$(i) bond!")
                    end
               end
          end

          obj = new(Vector{AbstractLocalOperator}(undef, L), strength * 1.0)
          for (i, Op) in enumerate(Ops)
               obj.Ops[i] = Op
          end
          return obj
     end

     function StringOperator(A::AbstractLocalOperator, Args...)
          if isa(Args[end], Number)
               return StringOperator(AbstractLocalOperator[A, Args[1:end-1]...], Args[end])
          else 
               return StringOperator(AbstractLocalOperator[A, Args...])
          end
     end
          
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
     return Ops
end

function reduce!(Ops::StringOperator)
     i = 1 
     while i < length(Ops)
          if Ops[i].si == Ops[i+1].si
               Ops[i] = Ops[i] * Ops[i+1]
               deleteat!(Ops, i+1)
          else
               i += 1
          end
     end
     return Ops
end