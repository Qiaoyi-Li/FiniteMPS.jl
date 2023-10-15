"""
     manualGC() -> nothing

Manually call `GC.gc()` on all workers and use `@timeit` to collect the time cost.
"""
function manualGC()
     @timeit GlobalTimer "manualGC" begin
          @everywhere GC.gc()
     end
     return nothing
end