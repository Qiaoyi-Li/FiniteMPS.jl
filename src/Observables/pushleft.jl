#     e
#     |
#  i--O2--a---
#     |       |
#     f       |
#     |       |
#  j--B---b --|
#     |       |
#     g       Er
#     |       |
#  k--O1--c --|
#     |       |
#     h       |
#     |       |
#  l--A---d---
#     |
#     e

function _pushleft(Er::BilayerRightTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, O2::LocalOperator{2, 1})::BilayerRightTensor{2, 1}
     return @tensor tmp[i j; l] := (((O1.A[h g] * B.A[j g f b]) * Er.A[b d]) * A.A[e d l h]) * O2.A[i f e]
end

function _pushleft(Er::BilayerRightTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, O2::LocalOperator{1, 1})::BilayerRightTensor{1, 1}
     return @tensor tmp[j; l] := (((O1.A[h g] * B.A[j g f b]) * Er.A[b d]) * A.A[e d l h]) * O2.A[f e]
end

function _pushleft(Er::BilayerRightTensor{2, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{1, 1}, B::MPSTensor{4}, ::IdentityOperator)::BilayerRightTensor{2, 1}
     return @tensor tmp[a j; l] := ((O1.A[h g] * B.A[j g e b]) * Er.A[a b d]) * A.A[e d l h]
end

function _pushleft(Er::BilayerRightTensor{1, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, O2::LocalOperator{2, 1})::BilayerRightTensor{2, 1}
     return @tensor tmp[i j; l] := ((B.A[j g f b] * Er.A[b d]) * O2.A[i f e])* A.A[e d l g]
end

function _pushleft(Er::BilayerRightTensor{1, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, ::IdentityOperator)::BilayerRightTensor{1, 1}
     return @tensor tmp[j; l] := (A.A[e d l g] * Er.A[b d]) * B.A[j g e b] 
end

function _pushleft(Er::BilayerRightTensor{2, 1}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, ::IdentityOperator)::BilayerRightTensor{2, 1}
     return @tensor tmp[a j; l] := (A.A[e d l g] * Er.A[a b d]) * B.A[j g e b] 
end

function _pushleft(Er::BilayerRightTensor{1, 1}, A::AdjointMPSTensor{4}, O1::LocalOperator{2, 1}, B::MPSTensor{4}, ::IdentityOperator)::BilayerRightTensor{1, 2}
     return @tensor tmp[j; k l] := ((A.A[e d l h] * Er.A[b d]) * O1.A[k h g]) * B.A[j g e b] 
end

function _pushleft(Er::BilayerRightTensor{1, 2}, A::AdjointMPSTensor{4}, ::IdentityOperator, B::MPSTensor{4}, ::IdentityOperator)::BilayerRightTensor{1, 2}
     return @tensor tmp[j; c l] := (A.A[e d l g] * Er.A[b c d]) * B.A[j g e b] 
end