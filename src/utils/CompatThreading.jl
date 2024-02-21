# some functions in TensorKit use multi-threading to parallelize the computation however seem to conflict with the high-level multi-threading implementations 
function _add_general_kernel_sequential!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)
     if iszero(β)
          zerovector!(tdst)
     elseif β != 1
          scale!(tdst, β)
     end
     for (f₁, f₂) in fusiontrees(tsrc)
          for ((f₁′, f₂′), coeff) in fusiontreetransform(f₁, f₂)
               TensorOperations.tensoradd!(tdst[f₁′, f₂′], p, tsrc[f₁, f₂], :N, α * coeff, true,
                    backend...)
          end
     end
     return nothing
end
function _add_abelian_kernel_sequential!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)

     for (f₁, f₂) in fusiontrees(tsrc)
          TensorKit._add_abelian_block!(tdst, tsrc, p, fusiontreetransform, f₁, f₂, α, β, backend...)
     end
     return tdst
end

TensorKit._add_abelian_kernel!(tdst::TensorMap{<:GradedSpace}, tsrc::TensorMap{<:GradedSpace}, p, fusiontreetransform, α, β, backend...) = _add_abelian_kernel_sequential!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)

TensorKit._add_general_kernel!(tdst::TensorMap{<:GradedSpace}, tsrc::TensorMap{<:GradedSpace}, p, fusiontreetransform, α, β, backend...) = _add_general_kernel_sequential!(tdst, tsrc, p, fusiontreetransform, α, β, backend...)

# multi-threading mul
using TensorKit: hasblock
function _blocktask_mul!(c, tA, tB, tC, α, β)
     if hasblock(tA, c)
          mul!(block(tC, c),
               block(tA, c),
               block(tB, c),
               α, β)
     elseif β != one(β)
          rmul!(block(tC, c), β)
     end
     return nothing
end

function _mul_atomic!(tC::AbstractTensorMap,
     tA::AbstractTensorMap,
     tB::AbstractTensorMap, α=true, β=false)
     if !(codomain(tC) == codomain(tA) && domain(tC) == domain(tB) &&
          domain(tA) == codomain(tB))
          throw(SpaceMismatch("$(space(tC)) ≠ $(space(tA)) * $(space(tB))"))
     end

     ntasks = get_num_threads_mul()
     if ntasks == 1 || !isa(blocksectors(tC), AbstractVector) || length(blocksectors(tC)) == 1
          for c in blocksectors(tC)
               _blocktask_mul!(c, tA, tB, tC, α, β)
          end
     else
          # sort sectors by size
          function blockcost(c)
               if hasblock(tA, c)
                    return size(block(tA, c), 1) * size(block(tA, c), 2) * size(block(tB, c), 2)
               else
                    return size(block(tC, c), 1) * size(block(tC, c), 2)
               end
          end
          sortedsectors = sort(blocksectors(tC); by=blockcost, rev=true)

          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:ntasks
               Threads.@spawn while true
                    i = Threads.atomic_add!(idx, 1)
                    i > length(sortedsectors) && break
                    _blocktask_mul!(sortedsectors[i], tA, tB, tC, α, β)
               end
          end

     end

     return tC
end

TensorKit.mul!(tC::AbstractTensorMap{<:GradedSpace}, tA::AbstractTensorMap{<:GradedSpace}, tB::AbstractTensorMap{<:GradedSpace}, α, β) = _mul_atomic!(tC, tA, tB, α, β)

# multi-threading svd
using TensorKit: SectorDict, MatrixAlgebra
function _blocktask_svd!(c, t, Udata, Σdata, Vdata, dims, alg)
     U, Σ, V = TensorKit.MatrixAlgebra.svd!(block(t, c), alg)
     Udata[c] = U
     Vdata[c] = V
     Σdata[c] = Σ
     dims[c] = length(Σ)
     return nothing
end

function _compute_svddata_atomic!(t::TensorMap, alg::Union{SVD,SDD})
     I = sectortype(t)
     c, b = first(blocks(t))
     emptymat = similar(b, (0, 0))
     emptyvec = similar(b, real(scalartype(b)), (0,))
     Udata = SectorDict(c => emptymat for c in blocksectors(t))
     Vdata = SectorDict(c => emptymat for c in blocksectors(t))
     Σdata = SectorDict(c => emptyvec for c in blocksectors(t))
     dims = SectorDict{I,Int}(c => 0 for c in blocksectors(t))

     ntasks = get_num_threads_svd()
     if ntasks == 1 || !isa(blocksectors(t), AbstractVector) || length(blocksectors(t)) == 1
          for c in blocksectors(t)
               _blocktask_svd!(c, t, Udata, Σdata, Vdata, dims, alg)
          end
     else
          # sort sectors by size
          function blockcost(c)
               b = block(t, c)
               return min(size(b, 1)^2 * size(b, 2), size(b, 1) * size(b, 2)^2)
          end
          perms = sortperm(blocksectors(t); by=blockcost, rev=true)
          sortedsectors = blocksectors(t)[perms]

          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:ntasks
               Threads.@spawn while true
                    i = Threads.atomic_add!(idx, 1)
                    i > length(sortedsectors) && break

                    _blocktask_svd!(sortedsectors[i], t, Udata, Σdata, Vdata, dims, alg)
               end
          end
     end
     return Udata, Σdata, Vdata, dims
end

TensorKit._compute_svddata!(t::TensorMap{<:GradedSpace}, alg::Union{SVD,SDD}) = _compute_svddata_atomic!(t, alg)

# multi-threading eigh
using TensorKit:similarstoragetype, Diagonal
function _eigh_atomic!(t::TensorMap)
     InnerProductStyle(t) === EuclideanProduct() || throw_invalid_innerproduct(:eigh!)
     domain(t) == codomain(t) ||
          throw(SpaceMismatch("`eigh!` requires domain and codomain to be the same"))

     S = spacetype(t)
     I = sectortype(t)
     A = storagetype(t)
     Ar = similarstoragetype(t, real(scalartype(t)))

     ntasks = get_num_threads_eig()
     if ntasks == 1 || !isa(blocksectors(t), AbstractVector) ||
        length(blocksectors(t)) == 1
        Ddata = SectorDict{I,Ar}()
        Vdata = SectorDict{I,A}()
        dims = SectorDict{I,Int}()
          for (c, b) in blocks(t)
               values, vectors = MatrixAlgebra.eigh!(b)
               d = length(values)
               Ddata[c] = copyto!(similar(values, (d, d)), Diagonal(values))
               Vdata[c] = vectors
               dims[c] = d
          end

     else
          # preallocate
          _, b = first(blocks(t))
          emptymat = similar(b, (0, 0))
          Ddata = SectorDict{I,Ar}(c => emptymat for c in blocksectors(t))
          Vdata = SectorDict{I,A}(c => emptymat for c in blocksectors(t))
          dims = SectorDict{I,Int}(c => 0 for c in blocksectors(t))

          # sort sectors by size
          function blockcost(c)
               return size(block(t, c), 1)
          end
          perms = sortperm(blocksectors(t); by=blockcost, rev=true)
          sortedsectors = blocksectors(t)[perms]

          idx = Threads.Atomic{Int64}(1)
          Threads.@sync for _ in 1:ntasks
               Threads.@spawn while true
                    i = Threads.atomic_add!(idx, 1)
                    i > length(sortedsectors) && break

                    c = sortedsectors[i]
                    values, vectors = MatrixAlgebra.eigh!(block(t, c))
                    d = length(values)
                    Ddata[c] = copyto!(similar(values, (d, d)), Diagonal(values))
                    Vdata[c] = vectors
                    dims[c] = d
               end
          end
     end

     if length(domain(t)) == 1
          W = domain(t)[1]
     else
          W = S(dims)
     end
     return TensorMap(Ddata, W ← W), TensorMap(Vdata, domain(t) ← W)
end

TensorKit.eigh!(t::TensorMap{<:GradedSpace}) = _eigh_atomic!(t)