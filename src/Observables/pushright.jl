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

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 2}, B::MPSTensor{4}, ::IdentityOperator)::BilayerLeftTensor{2, 1}
     return @tensor tmp[i j; k] := ((El.A[a c] * A.A[e i a f]) * O1.A[f g j]) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, ::IdentityOperator)::BilayerLeftTensor{1, 1}
     return @tensor tmp[i; k] := (El.A[a c] * A.A[e i a f]) * B.A[c f e k]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, O2::LocalOperator{2, 1})::BilayerLeftTensor{1, 1}
     return @tensor tmp[i; k] := ((El.A[a b c] * (A.A[e i a f] * O1.A[f g])) * O2.A[b h e]) * B.A[c g h k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, O2::LocalOperator{1, 1})::BilayerLeftTensor{1, 1}
     return @tensor tmp[i; k] := (((El.A[a c] * A.A[e i a f]) * O1.A[f g]) * B.A[c g h k]) * O2.A[h e] 
end


function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, ::IdentityOperator)::BilayerLeftTensor{2, 1}
     return @tensor tmp[i b; k] := (El.A[a b c] * (A.A[e i a f] * O1.A[f g])) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{2, 1})::BilayerLeftTensor{1, 1}
     return @tensor tmp[i; k] := ((El.A[a b c] * A.A[e i a f]) * O2.A[b h e]) * B.A[c f h k]
end

function _pushright(El::BilayerLeftTensor{2, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, ::IdentityOperator)::BilayerLeftTensor{2, 1}
     return @tensor tmp[i b; k] := (El.A[a b c] * A.A[e i a f]) * B.A[c f e k]
end

function _pushright(El::BilayerLeftTensor{1, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{1, 2})::BilayerLeftTensor{1, 2}
     return @tensor tmp[i; k l] := ((El.A[a c] * B.A[c g h k]) * O2.A[h e l]) * A.A[e i a g]
end

function _pushright(El::BilayerLeftTensor{1, 2}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, ::IdentityOperator)::BilayerLeftTensor{1, 1}
     return @tensor tmp[i; k] := ((El.A[a c d] * A.A[e i a f] ) * O1.A[d f g]) * B.A[c g e k]
end

function _pushright(El::BilayerLeftTensor{1, 2}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, ::IdentityOperator)::BilayerLeftTensor{1, 2}
     return @tensor tmp[i; k d] := (El.A[a c d] * A.A[e i a f] ) * B.A[c f e k]
end
