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
     V, ~, info = _truncS(x -> sqrt(x), V, rmul!(S2, 1 / Norm2), trunc)
     # MC = USV' => US = MCV, V -> CV if C is given
     if !isnothing(BondTensor)
          V = C * V
     end

     US_Al = Vector{MPSTensor}(undef, N)
     US_El = SparseLeftTensor(undef, N)
     if get_num_workers() > 1 # multi-processing
     # TODO
     else
          let V_wrap::MPSTensor = V
               @floop GlobalThreadsExecutor for i in 1:N
                    US_Al[i] = LO.Al[i] * V_wrap
                    US_El[i] = LO.El[i] * V_wrap
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
     U, ~, info = _truncS(x -> sqrt(x), U, rmul!(S2, 1 / Norm2), trunc)
     # CM = USV' => SV' = U'CM = (C'U)'M, U -> C'U if C is given
     if !isnothing(BondTensor)
          U = C' * U
     end

     SVd_Ar = Vector{MPSTensor}(undef, N)
     SVd_Er = SparseRightTensor(undef, N)
     if get_num_workers() > 1 # multi-processing
     # TODO
     else
          let U_wrap::MPSTensor = U'
               @floop GlobalThreadsExecutor for i in 1:N
                    SVd_Ar[i] = U_wrap * RO.Ar[i]
                    SVd_Er[i] = U_wrap * RO.Er[i]
               end
          end
     end

     return RightOrthComplement(SVd_Er, SVd_Ar, RO.Ar_c), info
end

function _CBE_MM(f, lsA::Vector{MPSTensor}, lsE::Union{SparseLeftTensor,SparseRightTensor})

     if get_num_workers() > 1 # multi-processing
     # TODO

     else
          @floop GlobalThreadsExecutor for (A, E) in zip(lsA, lsE)
               tmp = f(A) - f(E)
               @reduce() do (MM = nothing; tmp)
                    MM = axpy!(true, tmp, MM)
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