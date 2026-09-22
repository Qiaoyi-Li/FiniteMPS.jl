#        e
#        |
#  ---d--O2--l
# |      |     
# |      h     
# |      |     
# |---c--B---k 
# |      |       
# El     g      
# |      |      
# |---b--O1--j 
# |      |       
# |      f      
# |      |      
#  ---a--A---i
#        |
#        e

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 2}, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{2, 1}
	return @tensor allocator = allocator tmp[i j; k] := ((El.A[a c] * A.A[e i a f]) * O1.A[f g j]) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{1, 1}
	return @tensor allocator = allocator tmp[i; k] := (El.A[a c] * A.A[e i a f]) * B.A[c f e k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, O2::LocalOperator{1, 1}; allocator = ManualAllocator())::BilayerLeftTensor{1, 1}
	return @tensor allocator = allocator tmp[i; k] := (((El.A[a c] * A.A[e i a f]) * O1.A[f g]) * B.A[c g h k]) * O2.A[h e]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{2, 1}
	return @tensor allocator = allocator tmp[i b; k] := (El.A[a b c] * (A.A[e i a f] * O1.A[f g])) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{2, 1}
	return @tensor allocator = allocator tmp[i b; k] := (El.A[a b c] * A.A[e i a f]) * B.A[c f e k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{1, 2}; allocator = ManualAllocator())::BilayerLeftTensor{1, 2}
	return @tensor allocator = allocator tmp[i; k l] := ((El.A[a c] * B.A[c g h k]) * O2.A[h e l]) * A.A[e i a g]
end

function _pushright(El::BilayerLeftTensor{1, 2}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{1, 2}
	return @tensor allocator = allocator tmp[i; k d] := (El.A[a c d] * A.A[e i a f]) * B.A[c f e k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{1, 2}
	return @tensor allocator = allocator tmp[i; k b] := ((El.A[a c] * A.A[e i a f]) * B.A[c g e k]) * O1.A[b f g]
end

function _pushright(El::BilayerLeftTensor{1, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, O2::LocalOperator{2, 1}; allocator = ManualAllocator())::BilayerLeftTensor{1, 1}
	return @tensor allocator = allocator tmp[i; k] := ((El.A[a c d] * (A.A[e i a f] * O1.A[f g])) * O2.A[d h e]) * B.A[c g h k]
end

function _pushright(El::BilayerLeftTensor{1, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{1, 2}
	return @tensor allocator = allocator tmp[i; k d] := (El.A[a c d] * (A.A[e i a f] * O1.A[f g])) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{1, 2}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{2, 1}; allocator = ManualAllocator())::BilayerLeftTensor{1, 1}
	return @tensor allocator = allocator tmp[i; k] := ((El.A[a c d] * A.A[e i a f]) * O2.A[d h e]) * B.A[c f h k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{1, 1}; allocator = ManualAllocator())::BilayerLeftTensor{1, 1}
	return @tensor allocator = allocator tmp[i; k] := (El.A[a c] * (A.A[e i a f] * O2.A[h e])) * B.A[c f h k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{1, 1}
	return @tensor allocator = allocator tmp[i; k] := (El.A[a c] * (A.A[e i a f] * O1.A[f g])) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{2, 1}; allocator = ManualAllocator())::BilayerLeftTensor{2, 1}
	return @tensor allocator = allocator tmp[i d; k] := ((El.A[a c] * A.A[e i a f]) * O2.A[d h e]) * B.A[c f h k]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, O2::LocalOperator{1, 1}; allocator = ManualAllocator())::BilayerLeftTensor{1, 1}
	return @tensor allocator = allocator tmp[i; k] := ((El.A[a b c] * (B.A[c g h k] * O2.A[h e])) * O1.A[b f g]) * A.A[e i a f]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{1, 1}; allocator = ManualAllocator())::BilayerLeftTensor{2, 1}
	return @tensor allocator = allocator tmp[i b; k] := (El.A[a b c] * (A.A[e i a f] * O2.A[h e])) * B.A[c f h k]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{1, 1}
	return @tensor allocator = allocator tmp[i; k] := ((El.A[a b c] * A.A[e i a f]) * O1.A[b f g]) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, O2::LocalOperator{2, 1}; allocator = ManualAllocator())::BilayerLeftTensor{1, 1}
	return @tensor allocator = allocator tmp[i; k] := (((El.A[a c] * A.A[e i a f]) * O1.A[b f g]) * O2.A[b h e]) * B.A[c g h k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 2}, B::MPSTensor{4}, O2::LocalOperator{2, 2}; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i j; k l] := (((El.A[a c] * A.A[e i a f]) * O1.A[b f g j]) * B.A[c g h k]) * O2.A[b h e l]
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, O2::LocalOperator{2, 1}; allocator = ManualAllocator())::BilayerLeftTensor{1, 1}
	return @tensor allocator = allocator tmp[i; k] := (((El.A[a b c d] * A.A[e i a f]) * O1.A[b f g]) * O2.A[d h e]) * B.A[c g h k]
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, O2::LocalOperator{1, 1}; allocator = ManualAllocator())::BilayerLeftTensor{1, 2}
	return @tensor allocator = allocator tmp[i; k d] := ((El.A[a b c d] * A.A[e i a f]) * O1.A[b f g]) * (O2.A[h e] * B.A[c g h k])
end

function _pushright(El::BilayerLeftTensor{1, 2}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{1, 1}; allocator = ManualAllocator())::BilayerLeftTensor{1, 2}
	return @tensor allocator = allocator tmp[i; k d] := (El.A[a c d] * A.A[e i a f]) * (B.A[c f h k] * O2.A[h e])
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, O2::LocalOperator{2, 1}; allocator = ManualAllocator())::BilayerLeftTensor{2, 1}
	return @tensor allocator = allocator tmp[i b; k] := ((El.A[a b c d] * (A.A[e i a f] * O1.A[f g])) * O2.A[d h e]) * B.A[c g h k]
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, O2::LocalOperator{1, 1}; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i b; k d] := (El.A[a b c d] * (A.A[e i a f] * O1.A[f g])) * (O2.A[h e] * B.A[c g h k])
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 2}, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i j; k b] := ((El.A[a c] * A.A[e i a f]) * B.A[c g e k]) * O1.A[b f g j]
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, O2::LocalOperator{2, 2}; allocator = ManualAllocator())::BilayerLeftTensor{1, 2}
	return @tensor allocator = allocator tmp[i; k l] := (((El.A[a b c d] * A.A[e i a f]) * O1.A[b f g]) * B.A[c g h k]) * O2.A[d h e l]
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{1, 2}
	return @tensor allocator = allocator tmp[i; k d] := ((El.A[a b c d] * A.A[e i a f]) * O1.A[b f g]) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{1, 2}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{2, 2}; allocator = ManualAllocator())::BilayerLeftTensor{1, 2}
	return @tensor allocator = allocator tmp[i; k l] := (El.A[a c d] * A.A[e i a f]) * (B.A[c f h k] * O2.A[d h e l])
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{2, 2}; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i d; k l] := ((El.A[a c] * A.A[e i a f]) * B.A[c f h k]) * O2.A[d h e l]
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 2}, B::MPSTensor{4}, O2::LocalOperator{2, 1}; allocator = ManualAllocator())::BilayerLeftTensor{2, 1}
	return @tensor allocator = allocator tmp[i j; k] := (((El.A[a b c d] * O2.A[d h e]) * B.A[c g h k]) * O1.A[b f g j]) * A.A[e i a f]
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{2, 1}; allocator = ManualAllocator())::BilayerLeftTensor{2, 1}
	return @tensor allocator = allocator tmp[i b; k] := ((El.A[a b c d] * O2.A[d h e]) * B.A[c g h k]) * A.A[e i a g]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 2}, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{2, 1}
	return @tensor allocator = allocator tmp[i j; k] := ((El.A[a b c] * A.A[e i a f]) * O1.A[b f g j]) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 2}, B::MPSTensor{4}, O2::LocalOperator{1, 1}; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i j; k d] := ((El.A[a b c d] * A.A[e i a f]) * O1.A[b f g j]) * (O2.A[h e] * B.A[c g h k])
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{1, 1}; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i b; k d] := (El.A[a b c d] * A.A[e i a g]) * (O2.A[h e] * B.A[c g h k])
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, O2::LocalOperator{2, 2}; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i b; k l] := ((El.A[a b c d] * (A.A[e i a f] * O1.A[f g])) * B.A[c g h k]) * O2.A[d h e l]
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i b; k d] := (El.A[a b c d] * (A.A[e i a f] * O1.A[f g])) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 2}, B::MPSTensor{4}, O2::LocalOperator{1, 2}; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i j; k l] := (((El.A[a c] * A.A[e i a f]) * O1.A[f g j]) * B.A[c g h k]) * O2.A[h e l]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, O2::LocalOperator{1, 2}; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i b; k l] := ((El.A[a b c] * (A.A[e i a f] * O1.A[f g])) * B.A[c g h k]) * O2.A[h e l]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, O2::LocalOperator{1, 2}; allocator = ManualAllocator())::BilayerLeftTensor{1, 2}
	return @tensor allocator = allocator tmp[i; k l] := ((El.A[a b c] * (A.A[e i a f] * O1.A[b f g])) * B.A[c g h k]) * O2.A[h e l]
end

function _pushright(El::BilayerLeftTensor{1, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 2}, B::MPSTensor{4}, O2::LocalOperator{2, 1}; allocator = ManualAllocator())::BilayerLeftTensor{2, 1}
	return @tensor allocator = allocator tmp[i j; k] := (((El.A[a c d] * A.A[e i a f]) * O2.A[d h e]) * B.A[c g h k]) * O1.A[f g j]
end

function _pushright(El::BilayerLeftTensor{1, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 2}, B::MPSTensor{4}, O2::LocalOperator{1, 1}; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i j; k d] := (El.A[a c d] * (A.A[e i a f] * O2.A[h e]) * B.A[c g h k]) * O1.A[f g j]
end


function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{2, 2}; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i b; k l] := ((El.A[a b c d] * A.A[e i a g]) * B.A[c g h k]) * O2.A[d h e l]
end

function _pushright(El::BilayerLeftTensor{2, 2}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, ::IdentityOperator; allocator = ManualAllocator())::BilayerLeftTensor{2, 2}
	return @tensor allocator = allocator tmp[i b; k d] := (El.A[a b c d] * A.A[e i a g]) * B.A[c g e k]
end
