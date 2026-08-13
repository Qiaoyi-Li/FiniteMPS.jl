"""
	*(A::SparseMPO, B::SparseMPO) -> A*B::SparseMPO

Performs direct multiplication of two sparse MPOs. Warning: The resulting MPO may not be in a compact representation. 
"""
function *(A::SparseMPO{L}, B::SparseMPO{L}) where L 

     AB = Vector{SparseMPOTensor}(undef, L)
     for si in 1:L 
          AB[si] = _prod(A[si], B[si])
     end
	return SparseMPO(AB)

end

function _prod(A::SparseMPOTensor, B::SparseMPOTensor)
     
     mA, nA = size(A)
     mB, nB = size(B)

     AB = SparseMPOTensor(nothing, mA * mB, nA * nB)

     # manual kron
     for iA in 1:mA, iB in 1:mB 
          i = (iA - 1) * mB + iB
          for jA in 1:nA, jB in 1:nB
               isnothing(A[iA, jA]) && continue
               isnothing(B[iB, jB]) && continue

               j = (jA - 1) * nB + jB 
               AB[i, j] = _prod(A[iA, jA], B[iB, jB])

          end
     end
     return AB
end


#            f           
#            |
#        c --B--h 
#       /    |   \
#  a--isoL   e    isoR*-i      
#       \    |   /
#        b --A--g       
#            |
#            d 

function _prod(A::LocalOperator{2, 2}, B::LocalOperator{2, 2})
	@assert A.si == B.si

	isoL = isometry(fuse(codomain(A)[1], codomain(B)[1]), codomain(A)[1]⊗codomain(B)[1])
	isoR = isometry(fuse(domain(A)[2], domain(B)[2]), domain(A)[2]⊗domain(B)[2])
	@tensor AB[a d; f i] := isoL[a b c] * A.A[b d e g] * B.A[c e f h] * isoR'[g h i]

	# meta data
	fermionic = A.fermionic ⊻ B.fermionic
	aspace = (fuse(getLeftSpace(A), getLeftSpace(B)), fuse(getRightSpace(A), getRightSpace(B)))
	tag = ((A.tag[1][1]*B.tag[1][1], A.tag[1][2]), (B.tag[2][1], A.tag[2][2]*B.tag[2][2]))
	strength = A.strength[] * B.strength[]
	
	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{1, 2}, B::LocalOperator{2, 2})
	@assert A.si == B.si

	isoR = isometry(fuse(domain(A)[2], domain(B)[2]), domain(A)[2]⊗domain(B)[2])
	@tensor AB[c d; f i] := A.A[d e g] * B.A[c e f h] * isoR'[g h i]

	fermionic = A.fermionic ⊻ B.fermionic
	aspace = (getLeftSpace(B), fuse(getRightSpace(A), getRightSpace(B)))
	tag = ((B.tag[1][1], A.tag[1][1]), (B.tag[2][1], A.tag[2][2]*B.tag[2][2]))
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{2, 1}, B::LocalOperator{2, 2})
	@assert A.si == B.si

	isoL = isometry(fuse(codomain(A)[1], codomain(B)[1]), codomain(A)[1]⊗codomain(B)[1])
	@tensor AB[a d; f h] := isoL[a b c] * A.A[b d e] * B.A[c e f h]

	fermionic = A.fermionic ⊻ B.fermionic
	aspace = (fuse(getLeftSpace(A), getLeftSpace(B)), getRightSpace(B))
	tag = ((A.tag[1][1]*B.tag[1][1], A.tag[1][2]), (B.tag[2][1], B.tag[2][2]))
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{2, 2}, B::LocalOperator{1, 2})
	@assert A.si == B.si

	isoR = isometry(fuse(domain(A)[2], domain(B)[2]), domain(A)[2]⊗domain(B)[2])
	@tensor AB[b d; f i] := A.A[b d e g] * B.A[e f h] * isoR'[g h i]

	fermionic = A.fermionic ⊻ B.fermionic
	aspace = (getLeftSpace(A), fuse(getRightSpace(A), getRightSpace(B)))
	tag = ((A.tag[1][1], A.tag[1][2]), (B.tag[2][1], A.tag[2][2]*B.tag[2][2]))
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{2, 2}, B::LocalOperator{2, 1})
	@assert A.si == B.si

	isoL = isometry(fuse(codomain(A)[1], codomain(B)[1]), codomain(A)[1]⊗codomain(B)[1])
	@tensor AB[a d; f g] := isoL[a b c] * A.A[b d e g] * B.A[c e f]

	fermionic = A.fermionic ⊻ B.fermionic
	aspace = (fuse(getLeftSpace(A), getLeftSpace(B)), getRightSpace(A))
	tag = ((A.tag[1][1]*B.tag[1][1], A.tag[1][2]), (B.tag[2][1], A.tag[2][2]))
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{1, 2}, B::LocalOperator{1, 2})
	@assert A.si == B.si

	isoR = isometry(fuse(domain(A)[2], domain(B)[2]), domain(A)[2]⊗domain(B)[2])
	@tensor AB[d; f i] := A.A[d e g] * B.A[e f h] * isoR'[g h i]

	fermionic = A.fermionic ⊻ B.fermionic
	aspace = (getLeftSpace(A), fuse(getRightSpace(A), getRightSpace(B)))
	tag = ((A.tag[1][1],), (B.tag[2][1], A.tag[2][2]*B.tag[2][2]))
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{1, 2}, B::LocalOperator{2, 1})
	@assert A.si == B.si

	@tensor AB[c d; f g] := A.A[d e g] * B.A[c e f]

	fermionic = A.fermionic ⊻ B.fermionic
	aspace = (getLeftSpace(B), getRightSpace(A))
	tag = ((B.tag[1][1], A.tag[1][1]), (B.tag[2][1], A.tag[2][2]))
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{2, 1}, B::LocalOperator{1, 2})
	@assert A.si == B.si

	@tensor AB[b d; f h] := A.A[b d e] * B.A[e f h]

	fermionic = A.fermionic ⊻ B.fermionic
	aspace = (getLeftSpace(A), getRightSpace(B))
	tag = ((A.tag[1][1], A.tag[1][2]), (B.tag[2][1], B.tag[2][2]))
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{2, 1}, B::LocalOperator{2, 1})
	@assert A.si == B.si

	isoL = isometry(fuse(codomain(A)[1], codomain(B)[1]), codomain(A)[1]⊗codomain(B)[1])
	@tensor AB[a d; f] := isoL[a b c] * A.A[b d e] * B.A[c e f]

	fermionic = A.fermionic ⊻ B.fermionic
	aspace = (fuse(getLeftSpace(A), getLeftSpace(B)), getRightSpace(B))
	tag = ((A.tag[1][1]*B.tag[1][1], A.tag[1][2]), (B.tag[2][1],))
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{2, 2}, B::LocalOperator{1, 1})
	@assert A.si == B.si

	if istrivial(getLeftSpace(B))
		@tensor AB[b d; f g] := A.A[b d e g] * B.A[e f]
		aspace = (getLeftSpace(A), getRightSpace(A))
		tag = ((A.tag[1][1], A.tag[1][2]), (B.tag[2][1], A.tag[2][2]))
	else
		isoL = isometry(fuse(codomain(A)[1], getLeftSpace(B)), codomain(A)[1]⊗getLeftSpace(B))
		isoR = isometry(fuse(domain(A)[2], getRightSpace(B)), domain(A)[2]⊗getRightSpace(B))
		@tensor AB[a d; f i] := isoL[a b c] * A.A[b d e g] * B.A[e f] * isoR'[g c i]
		aspace = (fuse(getLeftSpace(A), getLeftSpace(B)), fuse(getRightSpace(A), getRightSpace(B)))
		tag = ((A.tag[1][1], A.tag[1][2]), (B.tag[2][1], A.tag[2][2]))
	end

	fermionic = A.fermionic ⊻ B.fermionic
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{1, 1}, B::LocalOperator{2, 2})
	@assert A.si == B.si

	if istrivial(getLeftSpace(A))
		@tensor AB[c d; f h] := A.A[d e] * B.A[c e f h]
		aspace = (getLeftSpace(B), getRightSpace(B))
		tag = ((B.tag[1][1], A.tag[1][1]), (B.tag[2][1], B.tag[2][2]))
	else
		isoL = isometry(fuse(getLeftSpace(A), codomain(B)[1]), getLeftSpace(A)⊗codomain(B)[1])
		isoR = isometry(fuse(getRightSpace(A), domain(B)[2]), getRightSpace(A)⊗domain(B)[2])
		@tensor AB[a d; f i] := isoL[a b c] * A.A[d e] * B.A[c e f h] * isoR'[b h i]
		aspace = (fuse(getLeftSpace(A), getLeftSpace(B)), fuse(getRightSpace(A), getRightSpace(B)))
		tag = ((B.tag[1][1], A.tag[1][1]), (B.tag[2][1], B.tag[2][2]))
	end

	fermionic = A.fermionic ⊻ B.fermionic
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{1, 1}, B::LocalOperator{1, 1})
	@assert A.si == B.si

	@tensor AB[d; f] := A.A[d e] * B.A[e f]
	lineA = !istrivial(getLeftSpace(A))
	lineB = !istrivial(getLeftSpace(B))
	if lineA && lineB
		aspace = (fuse(getLeftSpace(A), getLeftSpace(B)), fuse(getRightSpace(A), getRightSpace(B)))
	elseif lineA
		aspace = A.aspace
	elseif lineB
		aspace = B.aspace
	else
		aspace = (getLeftSpace(A), getRightSpace(B))
	end

	fermionic = A.fermionic ⊻ B.fermionic
	tag = ((A.tag[1][1],), (B.tag[2][1],))
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{1, 1}, B::LocalOperator{1, 2})
	@assert A.si == B.si

	if istrivial(getLeftSpace(A))
		@tensor AB[d; f h] := A.A[d e] * B.A[e f h]
		aspace = (getLeftSpace(A), getRightSpace(B))
		tag = ((A.tag[1][1],), (B.tag[2][1], B.tag[2][2]))
	else
		isoR = isometry(fuse(getRightSpace(A), domain(B)[2]), getRightSpace(A)⊗domain(B)[2])
		@tensor AB[c d; f i] := A.A[d e] * B.A[e f h] * isoR'[c h i]
		aspace = (getLeftSpace(A), fuse(getRightSpace(A), getRightSpace(B)))
		tag = (("", A.tag[1][1]), (B.tag[2][1], B.tag[2][2]))
	end

	fermionic = A.fermionic ⊻ B.fermionic
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{1, 2}, B::LocalOperator{1, 1})
	@assert A.si == B.si

	if istrivial(getLeftSpace(B))
		@tensor AB[d; f g] := A.A[d e g] * B.A[e f]
		aspace = (getLeftSpace(A), getRightSpace(A))
		tag = ((A.tag[1][1],), (B.tag[2][1], A.tag[2][2]))
	else
		isoR = isometry(fuse(domain(A)[2], getRightSpace(B)), domain(A)[2]⊗getRightSpace(B))
		@tensor AB[c d; f i] := A.A[d e g] * B.A[e f] * isoR'[g c i]
		aspace = (getLeftSpace(B), fuse(getRightSpace(A), getRightSpace(B)))
		tag = (("", A.tag[1][1]), (B.tag[2][1], A.tag[2][2]))
	end

	fermionic = A.fermionic ⊻ B.fermionic
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{1, 1}, B::LocalOperator{2, 1})
	@assert A.si == B.si

	if istrivial(getLeftSpace(A))
		@tensor AB[c d; f] := A.A[d e] * B.A[c e f]
		aspace = (getLeftSpace(B), getRightSpace(B))
		tag = ((B.tag[1][1], A.tag[1][1]), (B.tag[2][1],))
	else
		isoL = isometry(fuse(getLeftSpace(A), codomain(B)[1]), getLeftSpace(A)⊗codomain(B)[1])
		@tensor AB[a d; f c] := isoL[a c b] * A.A[d e] * B.A[b e f]
		aspace = (fuse(getLeftSpace(A), getLeftSpace(B)), getRightSpace(A))
		tag = ((B.tag[1][1], A.tag[1][1]), (B.tag[2][1], ""))
	end

	fermionic = A.fermionic ⊻ B.fermionic
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{2, 1}, B::LocalOperator{1, 1})
	@assert A.si == B.si

	if istrivial(getLeftSpace(B))
		@tensor AB[b d; f] := A.A[b d e] * B.A[e f]
		aspace = (getLeftSpace(A), getRightSpace(A))
		tag = ((A.tag[1][1], A.tag[1][2]), (B.tag[2][1],))
	else
		isoL = isometry(fuse(codomain(A)[1], getLeftSpace(B)), codomain(A)[1]⊗getLeftSpace(B))
		@tensor AB[a d; f c] := isoL[a b c] * A.A[b d e] * B.A[e f]
		aspace = (fuse(getLeftSpace(A), getLeftSpace(B)), getRightSpace(B))
		tag = ((A.tag[1][1], A.tag[1][2]), (B.tag[2][1], ""))
	end

	fermionic = A.fermionic ⊻ B.fermionic
	strength = A.strength[] * B.strength[]

	return LocalOperator(AB, A.name * B.name, A.si, fermionic, strength, tag; aspace = aspace)
end

# TODO: propagate the tag correctly after adding tag field to IdentityOperator in the future
function _prod(A::LocalOperator{2, 2}, B::IdentityOperator)
     @assert A.si == B.si

     fermionic = A.fermionic 
     tag = A.tag
     strength = A.strength[] * B.strength[]
     if istrivial(B.aspace)
          AB = A.A
          aspace = A.aspace
     else
          isoL = isometry(fuse(codomain(A)[1], B.aspace), codomain(A)[1]⊗B.aspace)
          isoR = isometry(fuse(domain(A)[2], B.aspace), domain(A)[2]⊗B.aspace)
          @tensor AB[a d; f i] := isoL[a b c] * A.A[b d f g] * isoR'[g c i]
          aspace = (fuse(getLeftSpace(A), B.aspace), fuse(getRightSpace(A), B.aspace))
     end

     return LocalOperator(AB, A.name, A.si, fermionic, strength, tag; aspace = aspace)

end

function _prod(A::IdentityOperator, B::LocalOperator{2, 2})
     @assert A.si == B.si

     fermionic = B.fermionic
     tag = B.tag
     strength = A.strength[] * B.strength[]
     if istrivial(A.aspace)
          AB = B.A
          aspace = B.aspace
     else
          isoL = isometry(fuse(A.aspace, codomain(B)[1]), A.aspace⊗codomain(B)[1])
          isoR = isometry(fuse(A.aspace, domain(B)[2]), A.aspace⊗domain(B)[2])
          @tensor AB[a d; f i] := isoL[a b c] * B.A[c d f h] * isoR'[b h i]
          aspace = (fuse(A.aspace, getLeftSpace(B)), fuse(A.aspace, getRightSpace(B)))
     end

     return LocalOperator(AB, B.name, B.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{1, 1}, B::IdentityOperator)
     @assert A.si == B.si

     AB = A.A
     tag = A.tag
     if istrivial(B.aspace)
          aspace = A.aspace
     elseif istrivial(getLeftSpace(A))
          aspace = (B.aspace, B.aspace)
     else
          aspace = (fuse(getLeftSpace(A), B.aspace), fuse(getRightSpace(A), B.aspace))
     end

     fermionic = A.fermionic
     strength = A.strength[] * B.strength[]

     return LocalOperator(AB, A.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::IdentityOperator, B::LocalOperator{1, 1})
     @assert A.si == B.si

     AB = B.A
     tag = B.tag
     if istrivial(A.aspace)
          aspace = B.aspace
     elseif istrivial(getLeftSpace(B))
          aspace = (A.aspace, A.aspace)
     else
          aspace = (fuse(A.aspace, getLeftSpace(B)), fuse(A.aspace, getRightSpace(B)))
     end

     fermionic = B.fermionic
     strength = A.strength[] * B.strength[]

     return LocalOperator(AB, B.name, B.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::IdentityOperator, B::IdentityOperator)
     @assert A.si == B.si

     if istrivial(A.aspace)
          aspace = B.aspace
     elseif istrivial(B.aspace)
          aspace = A.aspace
     else
          aspace = fuse(A.aspace, B.aspace)
     end

     return IdentityOperator(A.pspace, aspace, A.si, A.strength[] * B.strength[])
end

function _prod(A::LocalOperator{1, 2}, B::IdentityOperator)
     @assert A.si == B.si

     fermionic = A.fermionic
     if istrivial(B.aspace)
          AB = A.A
          tag = A.tag
     else
          isoR = isometry(fuse(domain(A)[2], B.aspace), domain(A)[2]⊗B.aspace)
          @tensor AB[c d; f i] := A.A[d f g] * isoR'[g c i]
          tag = (("", A.tag[1][1]), (A.tag[2][1], A.tag[2][2]))
     end
     strength = A.strength[] * B.strength[]
     aspace = (B.aspace, fuse(getRightSpace(A), B.aspace))

     return LocalOperator(AB, A.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::IdentityOperator, B::LocalOperator{1, 2})
     @assert A.si == B.si

     fermionic = B.fermionic
     if istrivial(A.aspace)
          AB = B.A
          tag = B.tag
     else
          isoR = isometry(fuse(A.aspace, domain(B)[2]), A.aspace⊗domain(B)[2])
          @tensor AB[b d; f i] := B.A[d f h] * isoR'[b h i]
          tag = (("", B.tag[1][1]), (B.tag[2][1], B.tag[2][2]))
     end
     strength = A.strength[] * B.strength[]
     aspace = (A.aspace, fuse(A.aspace, getRightSpace(B)))

     return LocalOperator(AB, B.name, B.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::LocalOperator{2, 1}, B::IdentityOperator)
     @assert A.si == B.si

     fermionic = A.fermionic
     if istrivial(B.aspace)
          AB = A.A
          tag = A.tag
     else
          isoL = isometry(fuse(codomain(A)[1], B.aspace), codomain(A)[1]⊗B.aspace)
          @tensor AB[a d; f c] := isoL[a b c] * A.A[b d f]
          tag = ((A.tag[1][1], A.tag[1][2]), (A.tag[2][1], ""))
     end
     strength = A.strength[] * B.strength[]
     aspace = (fuse(getLeftSpace(A), B.aspace), B.aspace)

     return LocalOperator(AB, A.name, A.si, fermionic, strength, tag; aspace = aspace)
end

function _prod(A::IdentityOperator, B::LocalOperator{2, 1})
     @assert A.si == B.si

     fermionic = B.fermionic
     if istrivial(A.aspace)
          AB = B.A
          tag = B.tag
     else
          isoL = isometry(fuse(A.aspace, codomain(B)[1]), A.aspace⊗codomain(B)[1])
          @tensor AB[a d; f b] := isoL[a b c] * B.A[c d f]
          tag = ((B.tag[1][1], B.tag[1][2]), (B.tag[2][1], ""))
     end
     strength = A.strength[] * B.strength[]
     aspace = (fuse(A.aspace, getLeftSpace(B)), A.aspace)

     return LocalOperator(AB, B.name, B.si, fermionic, strength, tag; aspace = aspace)
end
