"""
     manualGC([T::TimerOutput]) -> nothing

Manually call `GC.gc()` on all workers and use `@timeit` to collect the time cost if `T::TimerOutput` is provided.
"""
function manualGC(T::TimerOutput)
     @timeit T "manualGC" begin
          @everywhere GC.gc()
     end
     return nothing
end
function manualGC()
     @everywhere GC.gc()
     return nothing
end