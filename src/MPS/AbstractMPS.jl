"""
     abstract type AbstractMPS{L}

Abstract type of all MPS/MPO with length `L`.
"""
abstract type AbstractMPS{L} end

length(::AbstractMPS{L}) where L = L
for func in (:getindex, :lastindex, :setindex!, :iterate, :keys, :isassigned)
     @eval Base.$func(obj::AbstractMPS, args...) = $func(obj.A, args...)
end

"""
     issparse(::AbstractMPS) -> ::Bool

Check if a MPS/MPO object is sparse, e.g. `::MPS` -> `false`, `::SparseMPO` -> `true`.
"""
issparse(::AbstractMPS) = false

"""
     abstract type DenseMPS{L, T <:Union{Float64, ComplexF64}} <: AbstractMPS{L}

Abstract type of dense MPS/MPO with length `L`.
"""
abstract type DenseMPS{L, T <:Union{Float64, ComplexF64}} <: AbstractMPS{L} end

# promote local tensors
scalartype(::DenseMPS{L, T}) where {L, T} = T
function Base.setindex!(obj::DenseMPS{L, ComplexF64}, A::T, si::Int64) where {L, T <:Union{AbstractTensorMap, AbstractMPSTensor}}
     if scalartype(A) != ComplexF64
          return setindex!(obj.A, A*one(ComplexF64), si)
     else
          return setindex!(obj.A, A, si)
     end
end

"""  
     normalize!(obj::DenseMPS) -> obj

Normalize a given MPS according to inner-induced norm. 

Note we assume the MPS satisfies a canonical form and the center tensor is normalized, hence we only normalize `c`.
"""
function normalize!(obj::DenseMPS) 
     obj.c /= norm(obj)
     return obj
end

"""
     norm(obj::DenseMPS) -> ::Float64

Return the inner-induced norm. Note we assume the MPS satisfies a canonical form and the center tensor is normalized, hence the norm is just `abs(c)`.
"""
function norm(obj::DenseMPS)
     return abs(obj.c)
end

"""
     coef(obj::DenseMPS) -> ::F

Interface of `DenseMPS`, return the global coefficient, where `F` is the number type of given MPS.
"""
coef(obj::DenseMPS) = obj.c

"""
     Center(obj::DenseMPS) -> Vector (length 2)

Interface of `DenseMPS`, return the info of canonical center. `[a, b]` means left-canonical from `1` to `a-1` and right-canonical from `b+1` to `L`.
"""
Center(obj::DenseMPS) = obj.Center

"""
     complex(obj::DenseMPS{L}) -> ::DenseMPS{L, ComplexF64}

Return a copy of given MPS but with `ComplexF64` as basic field.
"""
function complex(obj::DenseMPS{L, Float64}) where L
     obj_c = similar(ComplexF64, obj)
     obj_c.c = obj.c
     obj_c.Center[:] = obj.Center[:]
     for i = 1:L
          obj_c[i] = obj[i]
     end
     return obj_c
end
complex(obj::DenseMPS{L, ComplexF64}) where L = deepcopy(obj)

function Base.show(io::IO, obj::DenseMPS{L}) where L
     any(i -> !isassigned(obj.A, i), 1:L) && return println(io, "$(typeof(obj)): L = $L, to be initialized !")

     # avoid to show bond info
     memory = obj |> Base.summarysize |> Base.format_bytes
     println(io, "$(typeof(obj)): Center = $(Center(obj)), Norm = $(norm(obj)), total memory = $memory")
     # bond dimenson
     lsi = ceil(Int64, log10(L)) # length of si to be printed
     for si = 1:L
          local A = obj[si]
          D, DD = dim(A, 1)
          println(io, "Bond ", lpad(si-1, lsi), "->", lpad(si, lsi), ": $(codomain(A).spaces[1]), dim = $(D) -> $(DD)")
     end
end