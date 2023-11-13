function _CBE_MMd2US(lsEr::SparseRightTensor;
     trunc::TruncationScheme=TensorKit.NoTruncation(),
     normalize::Bool=false)
     # M = USV' => MM' = U S^2 U', return U*S, svdinfo
     # contract c index and manually sum over b index
     #   a--
     #      |
     #   b--Er  --> a--Er*Er'--a'
     #      |
     #   c--    
     function _CBE_MM(Er::LocalRightTensor{2})
          return Er.A * Er.A'
     end
     function _CBE_MM(Er::LocalRightTensor{3})
          @tensor tmp[a; d] := Er.A[a b c] * Er.A'[c d b]
          return tmp
     end
     _CBE_MM(::Nothing) = nothing

     if get_num_workers() > 1 # multi-processing
     # TODO


     else # multi-threading
          @floop GlobalThreadsExecutor for Er in lsEr
               tmp = _CBE_MM(Er)
               @reduce() do (MMd = nothing; tmp)
                    MMd = axpy!(true, tmp, MMd)
               end
          end
     end

     # M = USV' -> MM' = U S^2 U'
     S2, U = eigh(MMd)
     # norm(S) = norm(M) = sqrt(tr(MM'))
     normalize && rmul!(S2, 1 / tr(MMd))
     U, S, info = _truncS(x -> sqrt(x), U, S2, trunc)

     return U * S, info
end

function _CBE_MdM2US(lsEl::SparseLeftTensor;
     trunc::TruncationScheme=TensorKit.NoTruncation(),
     normalize::Bool=false)
     # M = USV' => M'M = V S^2 V', return U*S = M*V, svdinfo
     # note M*V is a sparse environment tensor
     # contract a index and manually sum over b index
     #    --c
     #   |
     #  El--b --> c'--El'*El--c
     #   |
     #    --a    
     function _CBE_MM(El::LocalLeftTensor{2})
          return El.A' * El.A
     end
     function _CBE_MM(El::LocalLeftTensor{3})
          @tensor tmp[d; c] := El.A'[b d a] * El.A[a b c]
          return tmp
     end
     _CBE_MM(::Nothing) = nothing

     if get_num_workers() > 1 # multi-processing
     # TODO


     else # multi-threading
          @floop GlobalThreadsExecutor for El in lsEl
               tmp = _CBE_MM(El)
               @reduce() do (MdM = nothing; tmp)
                    MdM = axpy!(true, tmp, MdM)
               end
          end
     end

     # M = USV' => M'M = V S^2 V'
     S2, V = eigh(MdM)
     # norm(S) = norm(M) = sqrt(tr(MM'))
     normalize && rmul!(S2, 1 / tr(MdM))
     V, ~, info = _truncS(x -> sqrt(x), V, S2, trunc)

     # M*V
     US = SparseLeftTensor(undef, length(lsEl))
     if get_num_workers() > 1 # multi-processing
     # TODO
     else
          @floop GlobalThreadsExecutor for (i, El) in enumerate(lsEl)
               US[i] = El * convert(MPSTensor, V)
          end
     end

     return US, info

end

function _CBE_MMd2U(lsEl::SparseLeftTensor;
     trunc::TruncationScheme=TensorKit.NoTruncation(),
     normalize::Bool=false)
     # M = USV' => MM' = U S^2 U', return U, svdinfo
     # contract c index and manually sum over b index
     #    --c
     #   |
     #  El--b --> a--El*El'--a'
     #   |
     #    --a  
     function _CBE_MM(El::LocalLeftTensor{2})
          return El.A * El.A'
     end
     function _CBE_MM(El::LocalLeftTensor{3})
          if rank(El, 1) == 1
               @tensor tmp[a; d] := El.A[a b c] * El.A'[b c d]
          else
               @tensor tmp[a; d] := El.A[a b c] * El.A'[c d b]
          end
          return tmp
     end
     _CBE_MM(::Nothing) = nothing

     if get_num_workers() > 1 # multi-processing
     # TODO
     else
          @floop GlobalThreadsExecutor for El in lsEl
               tmp = _CBE_MM(El)
               @reduce() do (MMd = nothing; tmp)
                    MMd = axpy!(true, tmp, MMd)
               end
          end
     end

     # M = USV' -> MM' = U S^2 U'
     S2, U = eigh(MMd)
     # norm(S) = norm(M) = sqrt(tr(MM'))
     normalize && rmul!(S2, 1 / tr(MMd))
     U, ~, info = _truncS(x -> sqrt(x), U, S2, trunc)

     return U, info
end

function _CBE_MdM2SVd(lsEl::SparseLeftTensor;
     trunc::TruncationScheme=TensorKit.NoTruncation(),
     normalize::Bool=false)
     # M = USV' => M'M = V S^2 V', return S*V', svdinfo
     # contract a index and manually sum over b index
     #    --c
     #   |
     #  El--b  --> c'--El'*El--c
     #   |
     #    --a   
     function _CBE_MM(El::LocalLeftTensor{2})
          return El.A' * El.A
     end
     function _CBE_MM(El::LocalLeftTensor{3})
          @tensor tmp[d; c] := El.A'[b d a] * El.A[a b c]
          return tmp
     end
     _CBE_MM(::Nothing) = nothing

     if get_num_workers() > 1 # multi-processing
     # TODO

     else # multi-threading
          @floop GlobalThreadsExecutor for El in lsEl
               tmp = _CBE_MM(El)
               @reduce() do (MdM = nothing; tmp)
                    MdM = axpy!(true, tmp, MdM)
               end
          end
     end

     # M = USV' => M'M = V S^2 V'
     S2, V = eigh(MdM)
     # norm(S) = norm(M) = sqrt(tr(MM'))
     normalize && rmul!(S2, 1 / tr(MdM))
     V, S, info = _truncS(x -> sqrt(x), V, S2, trunc)

     return S * V', info

end

function _CBE_MMd2SVd(lsEr::SparseRightTensor;
     trunc::TruncationScheme=TensorKit.NoTruncation(),
     normalize::Bool=false)
     # M = USV' => MM' = U S^2 U', return S*V' = U'*M, svdinfo
     # note U'*M is a sparse environment tensor
     # contract c index and manually sum over b index
     #    a-- 
     #       |
     #    b--Er --> a--Er*Er'--a'
     #       |
     #    c--
     function _CBE_MM(Er::LocalRightTensor{2})
          return Er.A * Er.A'
     end
     function _CBE_MM(Er::LocalRightTensor{3})
          if rank(Er, 1) == 1
               @tensor tmp[a; d] := Er.A[a b c] * Er.A'[b c d]
          else
               @tensor tmp[a; d] := Er.A[a b c] * Er.A'[c d b]
          end
          return tmp
     end
     _CBE_MM(::Nothing) = nothing

     if get_num_workers() > 1 # multi-processing
     # TODO

     else # multi-threading
          @floop GlobalThreadsExecutor for Er in lsEr
               tmp = _CBE_MM(Er)
               @reduce() do (MMd = nothing; tmp)
                    MMd = axpy!(true, tmp, MMd)
               end
          end
     end

     # M = USV' => MM' = U S^2 U'
     S2, U = eigh(MMd)
     # norm(S) = norm(M) = sqrt(tr(MM'))
     normalize && rmul!(S2, 1 / tr(MMd))
     U, ~, info = _truncS(x -> sqrt(x), U, S2, trunc)

     # U'*M
     SVd = SparseRightTensor(undef, length(lsEr))
     if get_num_workers() > 1 # multi-processing
     # TODO
     else
          @floop GlobalThreadsExecutor for (i, Er) in enumerate(lsEr)
               SVd[i] = convert(MPSTensor, U') * Er
          end
     end

     return SVd, info

end

function _CBE_MdM2Vd(lsEr::SparseRightTensor;
     trunc::TruncationScheme=TensorKit.NoTruncation(),
     normalize::Bool=false)
     # M = USV' => M'M = V S^2 V', return V', svdinfo
     # contract a index and manually sum over b index
     #    a--  
     #       |
     #    b--Er --> c'--Er'*Er--c
     #       |
     #    c--
     function _CBE_MM(Er::LocalRightTensor{2})
          return Er.A' * Er.A
     end
     function _CBE_MM(Er::LocalRightTensor{3})
          if rank(Er, 1) == 1
               @tensor tmp[d; c] := Er.A'[b d a] * Er.A[a b c]
          else
               @tensor tmp[d; c] := Er.A'[d a b] * Er.A[a b c]
          end
          return tmp
     end
     _CBE_MM(::Nothing) = nothing

     if get_num_workers() > 1 # multi-processing
     # TODO

     else # multi-threading
          @floop GlobalThreadsExecutor for Er in lsEr
               tmp = _CBE_MM(Er)
               @reduce() do (MdM = nothing; tmp)
                    MdM = axpy!(true, tmp, MdM)
               end
          end
     end

     # M = USV' => M'M = V S^2 V'
     S2, V = eigh(MdM)
     # norm(S) = norm(M) = sqrt(tr(MM'))
     normalize && rmul!(S2, 1 / tr(MdM))
     V, ~, info = _truncS(x -> sqrt(x), V, S2, trunc)

     return V', info

end

function _truncS(f, U::AbstractTensorMap, S::AbstractTensorMap, trunc::TruncationScheme)
     # truncate the singular values in S and apply f to S
     # return U, S, BondInfo

     isa(trunc, TensorKit.NoTruncation) && return U, S, BondInfo(dim(S, 2)..., 0.0, NaN)

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