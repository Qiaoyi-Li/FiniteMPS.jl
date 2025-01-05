"""
	absorb!(El::SparseLeftTensor, x::MPSTensor{2}) -> nothing
     absorb!(Er::SparseRightTensor, x::MPSTensor{2}) -> nothing

Absorb a rank-2 bond tensor to a environment tensor.
"""
function absorb!(El::SparseLeftTensor, S::MPSTensor{2})

	validIdx = findall(x -> !isnothing(El[x]), 1:length(El))
	if get_num_workers() > 1
		@assert false "not implemented"
	elseif get_num_threads_julia() > 0 # multi-threading
		Threads.@threads :greedy for idx in validIdx
			El[idx] = _absorb(El[idx], S)
		end
	else
          for idx in validIdx
               El[idx] = _absorb(El[idx], S)
          end
	end
     return nothing
end

function absorb!(Er::SparseRightTensor, S::MPSTensor{2})

	validIdx = findall(x -> !isnothing(Er[x]), 1:length(Er))
	if get_num_workers() > 1
		@assert false "not implemented"
	elseif get_num_threads_julia() > 0 # multi-threading
		Threads.@threads :greedy for idx in validIdx
			Er[idx] = _absorb(Er[idx], S)
		end
	else
          for idx in validIdx
               Er[idx] = _absorb(Er[idx], S)
          end
	end
     return nothing
end

function _absorb(El::LocalLeftTensor{2}, S::MPSTensor{2})
     return S.A' * El.A * S.A
end
function _absorb(El::LocalLeftTensor{3}, S::MPSTensor{2})
     return @tensor allocator = ManualAllocator() tmp[a; c e] := S.A'[a b] * El.A[b c d] * S.A[d e]
end

function _absorb(Er::LocalRightTensor{2}, S::MPSTensor{2})
     return S.A * Er.A * S.A'
end
function _absorb(Er::LocalRightTensor{3}, S::MPSTensor{2})
     return @tensor allocator = ManualAllocator() tmp[a c; e] := S.A[a b] * Er.A[b c d] * S.A'[d e]
end
