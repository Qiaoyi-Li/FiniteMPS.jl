function _leftProj(A::MPSTensor{4}, Al::MPSTensor{4})
     @tensor tmp[g h; i e] := (A.A[a b c e] * Al.A'[c f a b]) * Al.A[g h i f]
     return tmp
end
function _leftProj(A::MPSTensor{3}, Al::MPSTensor{3})
     @tensor tmp[g h; e] := (A.A[a b e] * Al.A'[f a b]) * Al.A[g h f]
     return tmp
end
function _leftProj(A::MPSTensor{5}, Al::MPSTensor{4})
     @tensor tmp[g h; i d e] := (A.A[a b c d e] * Al.A'[c f a b]) * Al.A[g h i f]
     return tmp
end
function _leftProj(A::MPSTensor{4}, Al::MPSTensor{3})
     @tensor tmp[g h; d e] := (A.A[a b d e] * Al.A'[f a b]) * Al.A[g h f]
     return tmp
end
function _leftProj(El::LocalLeftTensor{2}, Al::MPSTensor{4})
     @tensor tmp[d e; f c] := El.A[a c] * Al.A[d e f a]
     return tmp
end
function _leftProj(El::LocalLeftTensor{2}, Al::MPSTensor{3})
     @tensor tmp[d e; c] := El.A[a c] * Al.A[d e a]
     return tmp
end
function _leftProj(El::LocalLeftTensor{3}, Al::MPSTensor{4})
     @tensor tmp[d e; f b c] := El.A[a b c] * Al.A[d e f a]
     return tmp
end
function _leftProj(El::LocalLeftTensor{3}, Al::MPSTensor{3})
     @tensor tmp[d e; b c] := El.A[a b c] * Al.A[d e a]
     return tmp
end
_leftProj(::Nothing, ::MPSTensor) = nothing

function _rightProj(A::MPSTensor{4}, Ar::MPSTensor{4})
     @tensor tmp[a g; h i] := (A.A[a b c e] * Ar.A'[c e f b]) * Ar.A[f g h i]
     return tmp
end
function _rightProj(A::MPSTensor{3}, Ar::MPSTensor{3})
     @tensor tmp[a g; i] := (A.A[a b e] * Ar.A'[e f b]) * Ar.A[f g i]
     return tmp
end
function _rightProj(A::MPSTensor{5}, Ar::MPSTensor{4})
     @tensor tmp[a g; h d i] := (A.A[a b c d e] * Ar.A'[c e f b]) * Ar.A[f g h i]
     return tmp
end
function _rightProj(A::MPSTensor{4}, Ar::MPSTensor{3})
     @tensor tmp[a g; d i] := (A.A[a b d e] * Ar.A'[e f b]) * Ar.A[f g i]
     return tmp
end
function _rightProj(Er::LocalRightTensor{2}, Ar::MPSTensor{4})
     @tensor tmp[a d; e f] := Er.A[a c] * Ar.A[c d e f]
     return tmp
end
function _rightProj(Er::LocalRightTensor{2}, Ar::MPSTensor{3})
     @tensor tmp[a d; f] := Er.A[a c] * Ar.A[c d f]
     return tmp
end
function _rightProj(Er::LocalRightTensor{3}, Ar::MPSTensor{4})
     @tensor tmp[a d; e b f] := Er.A[a b c] * Ar.A[c d e f]
     return tmp
end
function _rightProj(Er::LocalRightTensor{3}, Ar::MPSTensor{3})
     @tensor tmp[a d; b f] := Er.A[a b c] * Ar.A[c d f]
     return tmp
end
_rightProj(::Nothing, ::MPSTensor) = nothing

function _directsum_Al(Al::MPSTensor{R}, Al_f::MPSTensor{R}) where {R}
     # Al ⊕ Al_f

     EmbA, EmbB = oplusEmbed(Al.A, Al_f.A, R)
     return permute(Al.A, (Tuple(1:R-1), (R,))) * EmbA + permute(Al_f.A, (Tuple(1:R-1), (R,))) * EmbB
end

function _directsum_Ar(Ar::MPSTensor{R}, Ar_f::MPSTensor{R}) where {R}
     # Ar ⊕ Ar_f

     EmbA, EmbB = oplusEmbed(Ar.A, Ar_f.A, 1)
     return EmbA * permute(Ar.A, ((1,), Tuple(2:R))) + EmbB * permute(Ar_f.A, ((1,), Tuple(2:R)))
end

function _expand_Ar(Al::MPSTensor{3}, Al_ex::MPSTensor{3}, Ar::MPSTensor{3})::MPSTensor{3}
     @tensor Ar_ex[d e; f] := (Al.A[a b c] * Al_ex.A'[d a b]) * Ar.A[c e f]
     return Ar_ex
end
function _expand_Ar(Al::MPSTensor{4}, Al_ex::MPSTensor{4}, Ar::MPSTensor{4})::MPSTensor{4}
     @tensor Ar_ex[e f; g h] := (Al.A[a b c d] * Al_ex.A'[c e a b]) * Ar.A[d f g h]
     return Ar_ex
end

function _expand_Al(Ar::MPSTensor{3}, Ar_ex::MPSTensor{3}, Al::MPSTensor{3})::MPSTensor{3}
     @tensor Al_ex[a b; h] := (Ar_ex.A'[f h e] * Ar.A[d e f]) * Al.A[a b d]
     return Al_ex
end    

function _expand_Al(Ar::MPSTensor{4}, Ar_ex::MPSTensor{4}, Al::MPSTensor{4})::MPSTensor{4}
     @tensor Al_ex[a b; c h] := (Ar_ex.A'[f g h e] * Ar.A[d e f g]) * Al.A[a b c d]
     return Al_ex
end    