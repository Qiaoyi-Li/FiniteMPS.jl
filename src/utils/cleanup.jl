"""
     cleanup!(d::SerializedElementArray, n::Int64) -> nothing
Use `rm` to cleanup the file corresponding to the `n`-th element of `d::SerializedElementArray` in disk manually.

     cleanup!(d::SerializedElementArray, lsn::AbstractArray{Int64}) -> nothing
Vector version of the above usage.

     cleanup!(d::SerializedElementArray) -> nothing
Cleanup all files of `d::SerializedElementArray` manually. Note `setindex!` cannot be applied to `d` after this operation.
"""
function cleanup!(d::SerializedElementArray, n::Int64)
     @assert n ≤ length(d)
     return rm("$(d.pathname)/$(n).bin"; force = true)
end
function cleanup!(d::SerializedElementArray, lsn::AbstractArray{Int64})
     for n in lsn
          cleanup!(d, n)
     end
     return nothing
end
function cleanup!(d::SerializedElementArray)
     return rm("$(d.pathname)"; force = true, recursive = true)
end

