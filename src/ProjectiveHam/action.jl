"""
     action(obj::SparseProjectiveHamiltonian{2}, x::CompositeMPSTensor{2,T}; kwargs...)
     action(obj::SparseProjectiveHamiltonian{1}, x::MPSTensor; kwargs...) 

Wrap `action2` and `action1` to use `@timeit` macro. Detailed usages please see `action2` and `action1`.
"""
function action(obj::SparseProjectiveHamiltonian{2}, x::CompositeMPSTensor{2,T}; kwargs...) where {T<:NTuple{2,MPSTensor}}
     @timeit GlobalTimer "action2" action2(obj, x; kwargs...)
end

function action(obj::SparseProjectiveHamiltonian{1}, x::MPSTensor; kwargs...) 
     @timeit GlobalTimer "action1" action1(obj, x; kwargs...)
end