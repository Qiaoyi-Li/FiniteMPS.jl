"""
     struct SparseMPO{L} <: AbstractMPS{L}
          A::Vector{SparseMPOTensor}
     end

Concrete type of sparse MPO. 

Note an instance of this type is usually a Hamiltonian,  which will not cost too much memory, therefore we always store the local tensors in memory.

# Constructors
     SparseMPO(A::AbstractVector{SparseMPOTensor})
"""
struct SparseMPO{L} <: AbstractMPS{L}
     A::Vector{SparseMPOTensor}
     function SparseMPO(A::AbstractVector{SparseMPOTensor})
          L = length(A)
          return new{L}(A)
     end
end

convert(::Type{<:SparseMPO}, A::Vector{SparseMPOTensor}) = SparseMPO(A)
issparse(::SparseMPO) = true

function show(io::IO, obj::SparseMPO{L}) where L

     memory = obj |> Base.summarysize |> Base.format_bytes
     println(io, "$(typeof(obj)): total memory = $memory")
      # bond dimenson
      lsi = ceil(Int64, log10(L)) # length of si to be printed
      for si in 1:L
          D, DD = dim(obj[si], 1)
          print(io, "Bond ", lpad(si-1, lsi), "->", lpad(si, lsi), ": ") 
          println(io, "$(sum(D)) -> $(sum(DD))")
     end
end

"""
     scalartype(obj::SparseMPO) -> Float64/ComplexF64

Return the scalar type of given `SparseMPO`. Note return `Float64` iff all local tensors are real. 
"""
function scalartype(obj::SparseMPO)
     for M in obj.A
          for T in M
               isnothing(T) && continue
               isa(T, IdentityOperator) && continue
               scalartype(T) <: Complex && return ComplexF64
          end
     end
     return Float64
end


# TODO SparseMPO to MPO
