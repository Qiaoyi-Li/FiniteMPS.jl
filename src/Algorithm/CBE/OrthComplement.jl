mutable struct LeftOrthComplement{N}
     El::SparseLeftTensor
     Al::Vector{MPSTensor}
     const Al_c::MPSTensor
     function LeftOrthComplement(El::SparseLeftTensor, Al::Vector{MPSTensor}, Al_c::MPSTensor)
          # directly construct
          N = length(El)
          return new{N}(El, Al, Al_c)
     end
     function LeftOrthComplement(El_i::SparseLeftTensor, Al_c::MPSTensor, Hl::SparseMPOTensor, Al_i::MPSTensor=Al_c)
          # initialize Al
          Al = _initialize_Al(El_i, Al_i, Hl)
          # initialize El
          El = _initialize_El(Al, Al_c)
          N = length(Al)
          return new{N}(El, Al, Al_c)
     end
end

mutable struct RightOrthComplement{N}
     Er::SparseRightTensor
     Ar::Vector{MPSTensor}
     const Ar_c::MPSTensor
     function RightOrthComplement(Er::SparseRightTensor, Ar::Vector{MPSTensor}, Ar_c::MPSTensor)
          # directly construct
          N = length(Er)
          return new{N}(Er, Ar, Ar_c)
     end
     function RightOrthComplement(Er_i::SparseRightTensor, Ar_c::MPSTensor, Hr::SparseMPOTensor, Ar_i::MPSTensor=Ar_c)
          # initialize Ar
          Ar = _initialize_Ar(Er_i, Ar_i, Hr)
          # initialize Er
          Er = _initialize_Er(Ar, Ar_c)
          N = length(Ar)
          return new{N}(Er, Ar, Ar_c)
     end
end

length(::LeftOrthComplement{N}) where {N} = N
length(::RightOrthComplement{N}) where {N} = N

function _initialize_Al(El_i::SparseLeftTensor, Al_i::MPSTensor, Hl::SparseMPOTensor)

     sz = size(Hl)
     Al = Vector{MPSTensor}(undef, sz[2])

     validIdx = filter!(x -> !isnothing(El_i[x[1]]) && !isnothing(Hl[x[1], x[2]]), [(i, j) for i in 1:sz[1] for j in 1:sz[2]])

     if get_num_workers() > 1
          lsAl = pmap(validIdx) do (i, j)
               _initialize_Al_single(El_i[i], Al_i, Hl[i, j])
          end

          # sum over i
          for (idx, (_, j)) in enumerate(validIdx)
               if !isassigned(Al, j)
                    Al[j] = lsAl[idx]
               else
                    axpy!(true, lsAl[idx], Al[j])
               end
          end

     else

          numthreads = Threads.nthreads()
          # producer
          taskref = Ref{Task}()
          ch = Channel{Tuple{Int64,Int64}}(; taskref=taskref, spawn=true) do ch
               for idx in vcat(validIdx, fill((0, 0), numthreads))
                    put!(ch, idx)
               end
          end

          # consumer
          Lock = Threads.ReentrantLock()
          tasks = map(1:numthreads) do _
               task = Threads.@spawn while true
                    (i, j) = take!(ch)
                    i == 0 && break

                    tmp = _initialize_Al_single(El_i[i], Al_i, Hl[i, j])

                    lock(Lock)
                    try
                         if !isassigned(Al, j)
                              Al[j] = tmp
                         else
                              axpy!(true, tmp, Al[j])
                         end
                    catch
                         rethrow()
                    finally
                         unlock(Lock)
                    end

               end
               errormonitor(task)
          end

          wait.(tasks)
          wait(taskref[])
     end


     return Al
end

function _initialize_Ar(Er_i::SparseRightTensor, Ar_i::MPSTensor, Hr::SparseMPOTensor)

     sz = size(Hr)
     Ar = Vector{MPSTensor}(undef, sz[1])

     validIdx = filter!(x -> !isnothing(Er_i[x[2]]) && !isnothing(Hr[x[1], x[2]]), [(i, j) for i in 1:sz[1] for j in 1:sz[2]])

     if get_num_workers() > 1
          lsAr = pmap(validIdx) do (i, j)
               _initialize_Ar_single(Er_i[j], Ar_i, Hr[i, j])
          end
          # sum over j
          for (idx, (i, j)) in enumerate(validIdx)
               if !isassigned(Ar, i)
                    Ar[i] = lsAr[idx]
               else
                    axpy!(true, lsAr[idx], Ar[i])
               end
          end

     else

          Lock = Threads.ReentrantLock()
          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true
                    idx_t = Threads.atomic_add!(idx, 1)
                    idx_t > length(validIdx) && break

                    (i, j) = validIdx[idx_t]
                    tmp = _initialize_Ar_single(Er_i[j], Ar_i, Hr[i, j])

                    lock(Lock)
                    try
                         if !isassigned(Ar, i)
                              Ar[i] = tmp
                         else
                              axpy!(true, tmp, Ar[i])
                         end
                    catch
                         rethrow()
                    finally
                         unlock(Lock)
                    end
               end
          end
     end

     return Ar
end

function _initialize_El(Al::Vector{MPSTensor}, Al_i::MPSTensor)::SparseLeftTensor

     sz = length(Al)

     if get_num_workers() > 1
          return pmap(Al) do Al
               _initialize_El_single(Al, Al_i)
          end
     else
          El = SparseLeftTensor(undef, sz)

          numthreads = Threads.nthreads()
          # producer
          taskref = Ref{Task}()
          ch = Channel{Int64}(; taskref=taskref, spawn=true) do ch
               for idx in vcat(1:sz, fill(0, numthreads))
                    put!(ch, idx)
               end
          end

          # consumer
          tasks = map(1:numthreads) do _
               task = Threads.@spawn while true
                    i = take!(ch)
                    i == 0 && break
                    El[i] = _initialize_El_single(Al[i], Al_i)
               end
               errormonitor(task)
          end

          wait.(tasks)
          wait(taskref[])

          return El
     end


end

function _initialize_Er(Ar::Vector{MPSTensor}, Ar_i::MPSTensor)::SparseRightTensor

     sz = length(Ar)

     if get_num_workers() > 1
          return pmap(Ar) do Ar
               _initialize_Er_single(Ar, Ar_i)
          end
     else
          Er = SparseRightTensor(undef, sz)
     
          idx = Threads.Atomic{Int64}(1)
          @sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true
                    i = Threads.atomic_add!(idx, 1)
                    i > sz && break
                    Er[i] = _initialize_Er_single(Ar[i], Ar_i)
               end
          end

          return Er
     end
end

# =================== contract El, Al and Hl ====================
#             e(3)
#             |
#    --- c ---Al--- f(5)     
#   |         |
#   |         d
#   |         |
#  El--- b ---Hl--- h(4)
#   |         |
#   |         g(2)
#   |
#    --- a(1)
function _initialize_Al_single(El::LocalLeftTensor{2}, Al::MPSTensor, Hl::IdentityOperator)::MPSTensor
     return rmul!(El * Al, Hl.strength)
end
function _initialize_Al_single(El::LocalLeftTensor{2}, Al::MPSTensor{4}, Hl::LocalOperator{1,1})::MPSTensor
     @tensor tmp[a g; e f] := Hl.strength * El.A[a c] * (Hl.A[g d] * Al.A[c d e f])
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{2}, Al::MPSTensor{3}, Hl::LocalOperator{1,1})::MPSTensor
     @tensor tmp[a g; f] := Hl.strength * El.A[a c] * (Hl.A[g d] * Al.A[c d f])
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{2}, Al::MPSTensor{4}, Hl::LocalOperator{1,2})::MPSTensor
     @tensor tmp[a g; e h f] := Hl.strength * (El.A[a c] * Al.A[c d e f]) * Hl.A[g d h]
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{2}, Al::MPSTensor{3}, Hl::LocalOperator{1,2})::MPSTensor
     @tensor tmp[a g; h f] := Hl.strength * (El.A[a c] * Al.A[c d f]) * Hl.A[g d h]
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{4}, Hl::IdentityOperator)::MPSTensor
     @tensor tmp[a d; e b f] := Hl.strength * El.A[a b c] * Al.A[c d e f]
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{3}, Hl::IdentityOperator)::MPSTensor
     @tensor tmp[a d; b f] := Hl.strength * El.A[a b c] * Al.A[c d f]
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{4}, Hl::LocalOperator{2,1})::MPSTensor
     @tensor tmp[a g; e f] := Hl.strength * (El.A[a b c] * Al.A[c d e f]) * Hl.A[b g d]
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{3}, Hl::LocalOperator{2,1})::MPSTensor
     @tensor tmp[a g; f] := Hl.strength * (El.A[a b c] * Al.A[c d f]) * Hl.A[b g d]
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{4}, Hl::LocalOperator{1,1})::MPSTensor
     @tensor tmp[a g; e b f] := Hl.strength * El.A[a b c] * (Hl.A[g d] * Al.A[c d e f])
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{3}, Hl::LocalOperator{1,1})::MPSTensor
     @tensor tmp[a g; b f] := Hl.strength * El.A[a b c] * (Hl.A[g d] * Al.A[c d f])
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{4}, Hl::LocalOperator{2,2})::MPSTensor
     @tensor tmp[a g; e h f] := Hl.strength * (El.A[a b c] * Al.A[c d e f]) * Hl.A[b g d h]
     return tmp
end
function _initialize_Al_single(El::LocalLeftTensor{3}, Al::MPSTensor{3}, Hl::LocalOperator{2,2})::MPSTensor
     @tensor tmp[a g; h f] := Hl.strength * (El.A[a b c] * Al.A[c d f]) * Hl.A[b g d h]
     return tmp
end
# ------------------------------------------------------

# =================== contract Er, Ar and Hr ====================
#          c(3)
#           |
#   a(1) ---Ar--- d ---    
#           |          |
#           b          |
#           |          |
#   e(4) ---Hr--- g ---Er
#           |          |
#          f(2)        |
#                      |
#              h(5) ---
function _initialize_Ar_single(Er::LocalRightTensor{2}, Ar::MPSTensor, Hr::IdentityOperator)::MPSTensor
     return rmul!(Ar * Er, Hr.strength)
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{4}, Hr::IdentityOperator)::MPSTensor
     @tensor tmp[a b; c g h] := Hr.strength * Ar.A[a b c d] * Er.A[d g h]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{3}, Hr::IdentityOperator)::MPSTensor
     @tensor tmp[a b; g h] := Hr.strength * Ar.A[a b d] * Er.A[d g h]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{2}, Ar::MPSTensor{4}, Hr::LocalOperator{1,1})::MPSTensor
     @tensor tmp[a f; c h] := Hr.strength * (Ar.A[a b c d] * Er.A[d h]) * Hr.A[f b]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{2}, Ar::MPSTensor{3}, Hr::LocalOperator{1,1})::MPSTensor
     @tensor tmp[a f; h] := Hr.strength * (Ar.A[a b d] * Er.A[d h]) * Hr.A[f b]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{2}, Ar::MPSTensor{4}, Hr::LocalOperator{2,1})::MPSTensor
     @tensor tmp[a f; c e h] := Hr.strength * (Ar.A[a b c d] * Er.A[d h]) * Hr.A[e f b]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{2}, Ar::MPSTensor{3}, Hr::LocalOperator{2,1})::MPSTensor
     @tensor tmp[a f; e h] := Hr.strength * (Ar.A[a b d] * Er.A[d h]) * Hr.A[e f b]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{4}, Hr::LocalOperator{1,1})::MPSTensor
     @tensor tmp[a f; c g h] := Hr.strength * (Hr.A[f b] * Ar.A[a b c d]) * Er.A[d g h]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{3}, Hr::LocalOperator{1,1})::MPSTensor
     @tensor tmp[a f; g h] := Hr.strength * (Hr.A[f b] * Ar.A[a b d]) * Er.A[d g h]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{4}, Hr::LocalOperator{1,2})::MPSTensor
     @tensor tmp[a f; c h] := Hr.strength * (Hr.A[f b g] * Ar.A[a b c d]) * Er.A[d g h]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{3}, Hr::LocalOperator{1,2})::MPSTensor
     @tensor tmp[a f; h] := Hr.strength * (Hr.A[f b g] * Ar.A[a b d]) * Er.A[d g h]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{4}, Hr::LocalOperator{2,2})::MPSTensor
     @tensor tmp[a f; c e h] := Hr.strength * (Ar.A[a b c d] * Er.A[d g h]) * Hr.A[e f b g]
     return tmp
end
function _initialize_Ar_single(Er::LocalRightTensor{3}, Ar::MPSTensor{3}, Hr::LocalOperator{2,2})::MPSTensor
     @tensor tmp[a f; e h] := Hr.strength * (Ar.A[a b d] * Er.A[d g h]) * Hr.A[e f b g]
     return tmp
end
# =================== contract Al and Al_i ====================
#      c
#      | 
#   a--Al--e(3) 
#      | \  
#      b  d(2)
#      |
#  a--Al_i'--f(1)
#      |
#      c
function _initialize_El_single(Al::MPSTensor{4}, Al_i::MPSTensor{4})::LocalLeftTensor
     @tensor tmp[f; e] := Al.A[a b c e] * Al_i.A'[c f a b]
     return tmp
end
function _initialize_El_single(Al::MPSTensor{3}, Al_i::MPSTensor{3})::LocalLeftTensor
     @tensor tmp[f; e] := Al.A[a b e] * Al_i.A'[f a b]
     return tmp
end
function _initialize_El_single(Al::MPSTensor{5}, Al_i::MPSTensor{4})::LocalLeftTensor
     @tensor tmp[f d; e] := Al.A[a b c d e] * Al_i.A'[c f a b]
     return tmp
end
function _initialize_El_single(Al::MPSTensor{4}, Al_i::MPSTensor{3})::LocalLeftTensor
     @tensor tmp[f d; e] := Al.A[a b d e] * Al_i.A'[f a b]
     return tmp
end
# --------------------------------------------------------------

# =================== contract Ar and Ar_i ====================
#         c
#         | 
#  a(1)---Ar---e 
#       / |   
#   d(2)  b  
#         |
#  f(3)--Ar_i'--e
#         |
#         c
function _initialize_Er_single(Ar::MPSTensor{4}, Ar_i::MPSTensor{4})::LocalRightTensor
     @tensor tmp[a; f] := Ar.A[a b c e] * Ar_i.A'[c e f b]
     return tmp
end
function _initialize_Er_single(Ar::MPSTensor{3}, Ar_i::MPSTensor{3})::LocalRightTensor
     @tensor tmp[a; f] := Ar.A[a b e] * Ar_i.A'[e f b]
     return tmp
end
function _initialize_Er_single(Ar::MPSTensor{5}, Ar_i::MPSTensor{4})::LocalRightTensor
     @tensor tmp[a; d f] := Ar.A[a b c d e] * Ar_i.A'[c e f b]
     return tmp
end
function _initialize_Er_single(Ar::MPSTensor{4}, Ar_i::MPSTensor{3})::LocalRightTensor
     @tensor tmp[a; d f] := Ar.A[a b d e] * Ar_i.A'[e f b]
     return tmp
end
# --------------------------------------------------------------
