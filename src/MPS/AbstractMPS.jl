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
     abstract type DenseMPS{L} <: AbstractMPS{L}

Abstract type of dense MPS/MPO with length `L`.
"""
abstract type DenseMPS{L} <: AbstractMPS{L} end


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