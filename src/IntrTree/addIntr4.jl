"""
	 addIntr4!(Root::InteractionTreeNode,
		  Op::NTuple{4,AbstractTensorMap},
		  si::NTuple{4,Int64},
		  strength::Number;
		  Obs::Bool = false,
		  Z::Union{Nothing,AbstractTensorMap} = nothing,
		  name::NTuple{4,Union{Symbol,String}} = (:A, :B, :C, :D)) -> nothing

	 addIntr4!(Tree::InteractionTree, args...) = addIntr4!(Tree.Root.children[1], args...)

Add a 4-site interaction `Op` at site `si` (4tuple) to a given interaction tree. If Z is given, assume each operator in tuple `Op` is ferminic operator and add Z automatically.

	 addIntr4!(Root::InteractionTreeNode,
		  A::LocalOperator,
		  B::LocalOperator,
		  C::LocalOperator,
		  D::LocalOperator,
		  strength::Number,
		  Z::Union{Nothing,AbstractTensorMap};
		  value = nothing) -> nothing

Expert version, each method finally reduces to this one. 

Note if there exist repeated si, it will recurse to `addIntr2!` or `addIntr3!`(TODO) automatically.
"""
function addIntr4!(Root::InteractionTreeNode, O::NTuple{4, AbstractTensorMap}, si::NTuple{4, Int64}, strength::Number;
	Obs::Bool = false,
	Z::Union{Nothing, AbstractTensorMap} = nothing,
	name::NTuple{4, Union{Symbol, String}} = (:A, :B, :C, :D),
	value = Obs ? si => prod(string.(name)) : nothing)

	# convert to string
	name = string.(name)

	strength == 0 && return nothing

	# fermi flag 
	Zflag = !isnothing(Z)

	(A, B, C, D) = map(1:4) do i
		LocalOperator(O[i], name[i], si[i])
	end

	# deal with the permutation 1<->2 and 3<->4
	if si[1] > si[2]
		A, B = _swap(A, B)
		si = (si[2], si[1], si[3], si[4])
		!isnothing(Z) && (strength *= -1)
	end
	if si[3] > si[4]
		C, D = _swap(C, D)
		si = (si[1], si[2], si[4], si[3])
		!isnothing(Z) && (strength *= -1)
	end

	_addtag!(A, B, C, D)

	# ============ i == j == k == l ===========
	if A.si == B.si == C.si == D.si
		return addIntr1!(Root, A * B * C * D, strength; value = value)
	end

	# ============ reduce to 2-site term ===========
	if A.si == B.si
		if B.si == C.si
			# now C.si < D.si must be true
			return addIntr2!(Root, A * B * C, D, strength, Z; value = value)
		elseif C.si == D.si
			AB = A * B
			CD = C * D
			if AB.si > CD.si
				# AB and CD must be bosonic 
				AB, CD = _swap(AB, CD)
			end
			return addIntr2!(Root, AB, CD, strength, nothing; value = value)
		elseif A.si == D.si
			#     D           D
			# C Z Z  -->      B 
			#     B           A
			#     A       C Z Z 
			C_L = _leftOp(C)
			ABD_R = _rightOp((A * B) * D)
			C_L, ABD_R = _reduceOp(C_L, ABD_R)
			return addIntr2!(Root, C_L, ABD_R, strength, Z; value = value)
		end
	end
	if A.si == C.si
		if B.si == D.si
			#          D
			#  C Z Z Z Z              -BD
			#          B   -->   AC   
			#  A Z Z Z Z
			!isnothing(Z) && (strength = -strength)
			AC = _leftOp(A * C)
			BD = _rightOp(B * D)
			AC, BD = _reduceOp(AC, BD)
			return addIntr2!(Root, AC, BD, strength, nothing; value = value)
		elseif A.si == D.si
			# D 
			# C     
			#     B 
			# A Z Z
			return addIntr2!(Root, A * (C * D), B, strength, Z; value = value)
		end
	end
	if B.si == C.si == D.si
		#     D
		#     C
		#     B 
		# A Z Z 
		return addIntr2!(Root, A, B * (C * D), strength, Z; value = value)
	end

	# ============ reduce to 3-site term ===========
	if A.si == B.si
		if A.si < C.si
			#        D
			#    C Z Z
			# B
			# A
			return addIntr3!(Root, A * B, C, D, strength, Z; fermionic = (false, Zflag, Zflag), value = value)
		else
			#     D          D
			# C Z Z      C Z Z
			#   B    or         B
			#   A               A
			AB = A * B
			Op3 = (AB.A, C.A, D.A)
			si3 = (AB.si, C.si, D.si)
			name3 = (AB.name, C.name, D.name)
			return addIntr3!(Root, Op3, si3, strength; Obs = Obs, Z = Z, fermionic = (false, Zflag, Zflag), name = name3, value = value)
		end
	end
	if A.si == C.si
		if B.si < D.si
			#          D
			#  C Z Z Z Z                  D
			#      B      -->   AC   -B Z Z  
			#  A Z Z  
			!isnothing(Z) && (strength = -strength)
			return addIntr3!(Root, _leftOp(A * C), B, D, strength, Z; fermionic = (false, Zflag, Zflag), value = value)
		else
			#      D                    
			#  C Z Z                    B
			#          B  -->   AC  D Z Z   
			#  A Z Z Z Z
			return addIntr3!(Root, _leftOp(A * C), D, _rightOp(B), strength, Z; fermionic = (false, Zflag, Zflag), value = value)
		end
	end
	if A.si == D.si
		# note: AZ = -ZA
		#     D                      B
		# C Z Z                D
		#         B  -->      -A
		#     A Z Z       C Z  Z  Z  Z
		!isnothing(Z) && (strength = -strength)
		AD = A * D
		Op3 = _reduceOp(C, AD, B)
		return addIntr3!(Root, Op3..., strength, Z; fermionic = (Zflag, false, Zflag), value = value)
	end
	if B.si == C.si
		#         D                
		#     C Z Z            BC    D 
		#     B       -->  A Z Z  Z  Z
		# A Z Z
		return addIntr3!(Root, A, B * C, D, strength, Z; fermionic = (Zflag, false, Zflag), value = value)
	end
	if B.si == D.si
		if A.si < C.si
			#         D
			#     C Z Z           C   -BD  
			#         B  -->  A Z Z 
			# A Z Z Z Z  
			!isnothing(Z) && (strength = -strength)
			return addIntr3!(Root, A, _leftOp(C), _rightOp(B * D), strength, Z; fermionic = (Zflag, Zflag, false), value = value)
		else
			# note AZ = -ZA
			#         D
			# C Z Z Z Z           -A   -BD         A   BD
			#         B  -->  C Z  Z       --> C Z Z  
			#     A Z Z                 
			return addIntr3!(Root, _leftOp(C), A, _rightOp(B * D), strength, Z; fermionic = (Zflag, Zflag, false), value = value)
		end
	end
	if C.si == D.si
		if B.si < C.si
			#        D 
			#        C 
			#      B 
			#  A Z Z 
			return addIntr3!(Root, A, B, C * D, strength, Z; fermionic = (Zflag, Zflag, false), value = value)
		else
			#    D        D
			#    C        C
			#      B  or        B
			#  A Z Z        A Z Z
			CD = C * D
			Op3 = (A.A, B.A, CD.A)
			si3 = (A.si, B.si, CD.si)
			name3 = (A.name, B.name, CD.name)
			return addIntr3!(Root, Op3, si3, strength; Obs = Obs, Z = Z, fermionic = (Zflag, Zflag, false), name = name3, value = value)
		end

	end
	# ----------------------------------------------
	if B.si < C.si
		#             D
		#         C Z Z 
		#     B 
		# A Z Z 
		return addIntr4!(Root, A, B, C, D, strength, Z; value = value)
	else
		if A.si < C.si
			if B.si < D.si
				#             D                    D
				#     C Z Z Z Z               -B Z Z 
				#         B       -->      C 
				# A Z Z Z Z            A Z Z 
				!isnothing(Z) && (strength = -strength)
				return addIntr4!(Root, A, _leftOp(C), _rightOp(B), D, strength, Z; value = value)
			else
				A.si < C.si && B.si > D.si
				#         D                       B
				#     C Z Z                   D Z Z  
				#             B   -->      C
				# A Z Z Z Z Z Z        A Z Z 
				return addIntr4!(Root, A, _leftOp(C), D, _rightOp(B), strength, Z; value = value)
			end
		elseif A.si < D.si
			if B.si < D.si
				#             D
				# C Z Z Z Z Z Z                     D                    D
				#         B       -->      -A  -B Z Z  -->      A    B Z Z  
				#     A Z Z            C Z  Z               C Z Z 
				return addIntr4!(Root, _leftOp(C), A, _rightOp(B), D, strength, Z; value = value)
			else
				#         D
				# C Z Z Z Z                        B               
				#             B   -->      -A  D Z Z  
				#     A Z Z Z Z        C Z  Z             
				!isnothing(Z) && (strength = -strength)
				return addIntr4!(Root, _leftOp(C), A, D, _rightOp(B), strength, Z; value = value)
			end
		else
			#      D
			#  C Z Z  
			#             B
			#         A Z Z 
			return addIntr4!(Root, _leftOp(C), D, A, _rightOp(B), strength, Z; value = value)
		end
	end

	@assert false # make sure all cases are considered

end

function addIntr4!(Root::InteractionTreeNode, A::LocalOperator, B::LocalOperator, C::LocalOperator, D::LocalOperator,
	strength::Number, Z::Union{Nothing, AbstractTensorMap};
	value = nothing)
	@assert A.si < B.si < C.si < D.si

	#             D
	#         C Z Z 
	#     B 
	# A Z Z 

	!isnothing(Z) && map(x -> _addZ!(x, Z), (B, D))

	current_node = Root
	si = 1
	pspace = getPhysSpace(A)
	while si < D.si

		if si == A.si
			Op_i = A
		elseif si == B.si
			Op_i = B
		elseif si == C.si
			Op_i = C
		elseif !isnothing(Z) && (A.si < si < B.si || C.si < si)
			Op_i = LocalOperator(Z, :Z, si)
		else
			Op_i = IdentityOperator(pspace, si)
		end

		idx = findfirst(current_node.children) do x
			x.Op ≠ Op_i && return false
			if hastag(x.Op) && hastag(Op_i)
				x.Op.tag ≠ Op_i.tag && return false
			end
			return true
		end
		if isnothing(idx)
			addchild!(current_node, Op_i)
			current_node = current_node.children[end]
		else
			current_node = current_node.children[idx]
			# replace the tag 
			hastag(current_node.Op) && (current_node.Op.tag = Op_i.tag)
		end
		si += 1
	end

	idx = findfirst(x -> x.Op == D, current_node.children)
	if isnothing(idx)
		addchild!(current_node, D, value)
		current_node.children[end].Op.strength = strength
	else
		if !isnothing(value)
			# observable
			push!(current_node.children[idx].value, value)
		end
		_update_strength!(current_node.children[idx], strength) && deleteat!(current_node.children, idx)
	end

	return nothing

end

_addtag!(::LocalOperator{1, 1}, ::LocalOperator{1, 1}, ::LocalOperator{1, 1}, ::LocalOperator{1, 1}) = nothing

function _addtag!(A::LocalOperator{1, 2}, B::LocalOperator{2, 2}, C::LocalOperator{2, 2}, D::LocalOperator{2, 1})
	name = map(x -> x.name, [A, B, C, D])
	for i ∈ 2:4 # make sure each name is unique
		if any(==(name[i]), view(name, 1:i-1))
			name[i] = name[i] * "$i"
		end
	end
	A.tag = (("phys",), ("phys", "$(name[1])<-$(name[2])"))
	B.tag = (("$(name[1])<-$(name[2])", "phys"), ("phys", "$(name[2])<-$(name[3])"))
	C.tag = (("$(name[2])<-$(name[3])", "phys"), ("phys", "$(name[3])<-$(name[4])"))
	D.tag = (("$(name[3])<-$(name[4])", "phys"), ("phys",))
	return nothing
end

function _addtag!(A::LocalOperator{1, 2}, B::LocalOperator{2, 1}, C::LocalOperator{1, 2}, D::LocalOperator{2, 1})
	name = map(x -> x.name, [A, B, C, D])
	for i ∈ 2:4 # make sure each name is unique
		if any(==(name[i]), view(name, 1:i-1))
			name[i] = name[i] * "$i"
		end
	end
	A.tag = (("phys",), ("phys", "$(name[1])<-$(name[2])"))
	B.tag = (("$(name[1])<-$(name[2])", "phys"), ("phys",))
	C.tag = (("phys",), ("phys", "$(name[3])<-$(name[4])"))
	D.tag = (("$(name[3])<-$(name[4])", "phys"), ("phys",))
	return nothing
end
