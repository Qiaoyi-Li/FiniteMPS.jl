function _CBE_rightorth_L(RO::RightOrthComplement{N};
     trunc::TruncationScheme=TensorKit.NoTruncation(),
     normalize::Bool=false) where {N}
     # M = USV' => MM' = U S^2 U', return U*S, svdinfo
     # note MM' = Ar*Ar' - Er*Er'
     # contract c index and manually sum over b index
     #   a--
     #      |
     #   b--Er  --> a--Er*Er'--a'
     #      |
     #   c--    
     function f(Er::LocalRightTensor{2})
          return Er.A * Er.A'
     end
     function f(Er::LocalRightTensor{3})
          if rank(Er, 1) == 1
               @tensor tmp[a; d] := Er.A[a b c] * Er.A'[b c d]
          else
               @tensor tmp[a; d] := Er.A[a b c] * Er.A'[c d b]
          end
          return tmp
     end
     #       c
     #       |     
     #    a--Ar--e -->  a--Ar*Ar'--a'  
     #     / |
     #    d  b  
     function f(Ar::MPSTensor{3})
          @tensor tmp[a; f] := Ar.A[a b e] * Ar.A'[e f b]
          return tmp
     end
     function f(Ar::MPSTensor{4})
          @tensor tmp[a; f] := Ar.A[a b c e] * Ar.A'[c e f b]
          return tmp
     end
     function f(Ar::MPSTensor{5})
          @tensor tmp[a; f] := Ar.A[a b c d e] * Ar.A'[c d e f b]
          return tmp
     end
     f(::Nothing) = nothing

     MM = _CBE_MM(f, RO.Ar, RO.Er)

     # M = USV' -> MM' = U S^2 U'
     S2, U = eigh(MM)
     # norm(S) = norm(M) = sqrt(tr(MM'))
     Norm2 = tr(MM)
     U, S, info = _truncS(x -> sqrt(x), U, rmul!(S2, 1 / Norm2), trunc)

     if normalize
          normalize!(S)
     else
          rmul!(S, sqrt(Norm2)) # give back the norm
     end
     return U * S, info
end

function _CBE_rightorth_L(LO::LeftOrthComplement{N};
     BondTensor::Union{Nothing,MPSTensor}=nothing,
     trunc::TruncationScheme=TensorKit.NoTruncation()) where {N}
     # M = USV' => M'M = V S^2 V', return U*S = M*V, svdinfo
     # note M*V is a length-N vector of MPSTensor 
     # M'M = Al'*Al - El'*El
     # contract a index and manually sum over b index
     #    --c
     #   |
     #  El--b --> c'--El'*El--c
     #   |
     #    --a    
     function f(El::LocalLeftTensor{2})
          return El.A' * El.A
     end
     function f(El::LocalLeftTensor{3})
          if rank(El, 1) == 1
               @tensor tmp[d; c] := El.A'[b d a] * El.A[a b c]
          else
               @tensor tmp[d; c] := El.A'[d a b] * El.A[a b c]
          end
          return tmp
     end
     #       c
     #       |     
     #    a--Al--e -->  e'--Al'*Al-e  
     #       | \
     #       b  d 
     function f(Al::MPSTensor{3})
          @tensor tmp[f; e] := Al.A'[f a b] * Al.A[a b e]
          return tmp
     end
     function f(Al::MPSTensor{4})
          @tensor tmp[f; e] := Al.A'[c f a b] * Al.A[a b c e]
          return tmp
     end
     function f(Al::MPSTensor{5})
          @tensor tmp[f; e] := Al.A'[c d f a b] * Al.A[a b c d e]
          return tmp
     end
     f(::Nothing) = nothing

     MM = _CBE_MM(f, LO.Al, LO.El)

     if !isnothing(BondTensor)
          C = BondTensor.A
          # M'M -> C'M'MC
          MM = C' * MM * C
     end

     # M = USV' => M'M = V S^2 V'
     S2, V = eigh(MM)

     # norm(S) = norm(M) = sqrt(tr(MM'))
     Norm2 = tr(MM)
     V, _, info = _truncS(x -> sqrt(x), V, rmul!(S2, 1 / Norm2), trunc)
     # MC = USV' => US = MCV, V -> CV if C is given
     if !isnothing(BondTensor)
          V = C * V
     end

     if get_num_workers() > 1 # multi-processing
          US_Al::Vector{MPSTensor}, US_El::SparseLeftTensor = let V_wrap::MPSTensor = V
               US_Al = pmap(LO.Al) do Al
                    Al * V_wrap
               end
               US_El = pmap(LO.El) do El
                    El * V_wrap
               end
               US_Al, US_El
          end
     else

          US_Al = Vector{MPSTensor}(undef, N)
          US_El = SparseLeftTensor(undef, N)
          V_wrap::MPSTensor = V
          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true
                    idx_t = Threads.atomic_add!(idx, 1)

                    if idx_t <= N
                         US_Al[idx_t] = LO.Al[idx_t] * V_wrap
                    elseif idx_t <= 2 * N
                         US_El[idx_t-N] = LO.El[idx_t-N] * V_wrap
                    else
                         break
                    end
               end
          end

     end

     return LeftOrthComplement(US_El, US_Al, LO.Al_c), info
end

function _CBE_leftorth_R(LO::LeftOrthComplement{N};
     trunc::TruncationScheme=TensorKit.NoTruncation(),
     normalize::Bool=false) where {N}
     # M = USV' => M'M = V S^2 V', return S*V', svdinfo
     # note M'M = Al'*Al - El'*El
     # contract a index and manually sum over b index
     #    --c
     #   |
     #  El--b  --> c'--El'*El--c
     #   |
     #    --a
     function f(El::LocalLeftTensor{2})
          return El.A' * El.A
     end
     function f(El::LocalLeftTensor{3})
          if rank(El, 1) == 1
               @tensor tmp[d; c] := El.A'[b d a] * El.A[a b c]
          else
               @tensor tmp[d; c] := El.A'[d a b] * El.A[a b c]
          end
          return tmp
     end
     #       c
     #       |
     #    a--Al--e -->  e'--Al'*Al-e
     #       | \
     #       b  d
     function f(Al::MPSTensor{3})
          @tensor tmp[f; e] := Al.A'[f a b] * Al.A[a b e]
          return tmp
     end
     function f(Al::MPSTensor{4})
          @tensor tmp[f; e] := Al.A'[c f a b] * Al.A[a b c e]
          return tmp
     end
     function f(Al::MPSTensor{5})
          @tensor tmp[f; e] := Al.A'[c d f a b] * Al.A[a b c d e]
          return tmp
     end
     f(::Nothing) = nothing

     MM = _CBE_MM(f, LO.Al, LO.El)

     # M = USV' => M'M = V S^2 V'
     S2, V = eigh(MM)
     # norm(S) = norm(M) = sqrt(tr(MM'))
     Norm2 = tr(MM)
     V, S, info = _truncS(x -> sqrt(x), V, rmul!(S2, 1 / Norm2), trunc)

     if normalize
          normalize!(S)
     else
          rmul!(S, sqrt(Norm2)) # give back the norm
     end
     return S * V', info
end

function _CBE_leftorth_R(RO::RightOrthComplement{N};
     BondTensor::Union{Nothing,MPSTensor}=nothing,
     trunc::TruncationScheme=TensorKit.NoTruncation()) where {N}
     # M = USV' => MM' = U S^2 U', return S*V' = U'*M, svdinfo
     # note U'*M is a length-N vector of MPSTensor
     # MM' = Ar*Ar' - Er*Er'
     # contract c index and manually sum over b index
     #   a--
     #      |
     #   b--Er  --> a--Er*Er'--a'
     #      |
     #   c--
     function f(Er::LocalRightTensor{2})
          return Er.A * Er.A'
     end
     function f(Er::LocalRightTensor{3})
          if rank(Er, 1) == 1
               @tensor tmp[a; d] := Er.A[a b c] * Er.A'[b c d]
          else
               @tensor tmp[a; d] := Er.A[a b c] * Er.A'[c d b]
          end
          return tmp
     end
     #       c
     #       |
     #    a--Ar--e -->  a--Ar*Ar'--a'
     #     / |
     #    d  b
     function f(Ar::MPSTensor{3})
          @tensor tmp[a; f] := Ar.A[a b e] * Ar.A'[e f b]
          return tmp
     end
     function f(Ar::MPSTensor{4})
          @tensor tmp[a; f] := Ar.A[a b c e] * Ar.A'[c e f b]
          return tmp
     end
     function f(Ar::MPSTensor{5})
          @tensor tmp[a; f] := Ar.A[a b c d e] * Ar.A'[c d e f b]
          return tmp
     end
     f(::Nothing) = nothing

     MM = _CBE_MM(f, RO.Ar, RO.Er)

     if !isnothing(BondTensor)
          C = BondTensor.A
          # MM' -> CMM'C'
          MM = C * MM * C'
     end

     # M = USV' => MM' = U S^2 U'
     S2, U = eigh(MM)
     # norm(S) = norm(M) = sqrt(tr(MM'))
     Norm2 = tr(MM)
     U, _, info = _truncS(x -> sqrt(x), U, rmul!(S2, 1 / Norm2), trunc)
     # CM = USV' => SV' = U'CM = (C'U)'M, U -> C'U if C is given
     if !isnothing(BondTensor)
          U = C' * U
     end

     if get_num_workers() > 1 # multi-processing
          SVd_Ar::Vector{MPSTensor}, SVd_Er::SparseRightTensor = let U_wrap::MPSTensor = U'
               SVd_Ar = pmap(RO.Ar) do Ar
                    U_wrap * Ar
               end
               SVd_Er = pmap(RO.Er) do Er
                    U_wrap * Er
               end
               SVd_Ar, SVd_Er
          end
     else

          SVd_Ar = Vector{MPSTensor}(undef, N)
          SVd_Er = SparseRightTensor(undef, N)
          U_wrap::MPSTensor = U'
          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true
                    idx_t = Threads.atomic_add!(idx, 1)

                    if idx_t <= N
                         SVd_Ar[idx_t] = U_wrap * RO.Ar[idx_t]
                    elseif idx_t <= 2 * N
                         SVd_Er[idx_t-N] = U_wrap * RO.Er[idx_t-N]
                    else
                         break
                    end
               end
          end


     end

     return RightOrthComplement(SVd_Er, SVd_Ar, RO.Ar_c), info
end

function _CBE_MM(f, lsA::Vector{MPSTensor}, lsE::Union{SparseLeftTensor,SparseRightTensor})

     if get_num_workers() > 1 # multi-processing
          MM = @distributed (add!) for i in 1:length(lsA)
               f(lsA[i]) - f(lsE[i])
          end

     else

          MM = nothing
          Lock = Threads.ReentrantLock()
          idx = Threads.Atomic{Int64}(1)
          N = length(lsA)
          Threads.@sync for _ in 1:Threads.nthreads()
               Threads.@spawn while true

                    idx_t = Threads.atomic_add!(idx, 1)
                    if idx_t <= N
                         tmp = f(lsA[idx_t])
                    elseif idx_t <= 2 * N
                         tmp = rmul!(f(lsE[idx_t-N]), -1)
                    else
                         break
                    end

                    lock(Lock)
                    try
                         MM = axpy!(true, tmp, MM)
                    catch
                         rethrow()
                    finally
                         unlock(Lock)
                    end
               end
          end
     end
     return MM
end

function _truncS(f, U::AbstractTensorMap, S::AbstractTensorMap, trunc::NoTruncation)
     f == identity && return U, S, BondInfo(S, 0.0)

     S = deepcopy(S)
     for c in blocksectors(S)
          blocks(S)[c] = diagm(f.(diag(blocks(S)[c])))
     end
     return U, S, BondInfo(S, 0.0)
end

function _truncS(f, U::AbstractTensorMap, S::AbstractTensorMap, trunc::TruncationScheme)
     # truncate the singular values in S and apply f to S
     # return U, S, BondInfo

     T = spacetype(S)
     I = sectortype(S)
     A = Vector{scalartype(S)}

     # diagonal of S
     Sdata = TensorKit.SectorDict{I,A}()
     perms = TensorKit.SectorDict{I,Vector{Int}}()
     for k in keys(blocks(S))
          perms[k] = sortperm(diag(blocks(S)[k]), rev=true) # sort in descending order
          Sdata[k] = diag(blocks(S)[k])[perms[k]]
     end

     Sdata, ϵ = TensorKit._truncate!(Sdata, trunc, 2)
     Udata = TensorKit.SectorDict{I,storagetype(U)}()
     Smdata = TensorKit.SectorDict{I,storagetype(S)}()
     truncdims = TensorKit.SectorDict{I,Int}()
     for c in blocksectors(S)
          truncdim = length(Sdata[c])
          truncdim == 0 && continue

          truncdims[c] = truncdim
          Udata[c] = view(data(U)[c], :, perms[c][1:truncdim])
          Smdata[c] = diagm(f.(Sdata[c]))
     end
     W = T(truncdims)

     S_f = TensorMap(Smdata, W, W)
     return TensorMap(Udata, codomain(U), W), S_f, BondInfo(S_f, ϵ)
end
_truncS(U::AbstractTensorMap, S::AbstractTensorMap, trunc::TruncationScheme) = _truncS(identity, U, S, trunc)