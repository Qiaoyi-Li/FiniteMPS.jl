function _preselect(LO::LeftOrthComplement{N}) where {N} 

     function f(El::LocalLeftTensor{2}, iso::AbstractTensorMap)
          return El.A * iso
     end
     function f(El::LocalLeftTensor{3}, iso::AbstractTensorMap)
          @tensor tmp[a; d] := El.A[a b c] * iso[b c d]
          return tmp
     end
     function f(Al::MPSTensor{4}, iso::AbstractTensorMap)
          if rank(iso) == 2
               @tensor tmp[a b; c f] := Al.A[a b c e] * iso[e f]
          else
               # MPS tensor with an additional bond
               @tensor tmp[a b; f] := Al.A[a b d e] * iso[d e f]
          end
          return tmp
     end
     function f(Al::MPSTensor{5}, iso::AbstractTensorMap)
          @tensor tmp[a b; c f] := Al.A[a b c d e] * iso[d e f]
          return tmp
     end
     f(::Nothing, ::AbstractTensorMap) = nothing

     # fuse the horizontal bonds of MPS and H
     lsIso = fuse(LO.El)

     if get_num_workers() > 1 # multi-processing
     #TODO
     else
          El, Al = let f = f
               @floop GlobalThreadsExecutor for i in 1:N
                    El_i = f(LO.El[i], lsIso[i])
                    Al_i = f(LO.Al[i], lsIso[i])
                    @reduce() do (El = nothing; El_i), (Al = nothing; Al_i)
                         El = axpy!(true, El_i, El)
                         Al = axpy!(true, Al_i, Al)  
                    end
               end
               if rank(Al, 2) > 1
                    Al = permute(Al, Tuple(1:rank(Al) - 1), (rank(Al),))
               end
               El, Al
          end
     end

     Al_c = let Al = LO.Al_c.A
          if rank(Al, 2) > 1
               Al = permute(Al, Tuple(1:rank(Al) - 1), (rank(Al),))
          end
          Al
     end
     # project to the orthogonal complement 
     Al_oc = normalize!(Al - Al_c * El) 
     
     return Al_oc
end