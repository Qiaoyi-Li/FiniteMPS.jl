"""
     trivial(V::VectorSpace) -> ::VectorSpace

Return the trivial space of a given vector space.

# Examples
```julia
julia> trivial(Rep[U₁ × SU₂]((1, 1/2) => 2))
Rep[U₁ × SU₂]((0, 0)=>1)

julia> trivial(ℂ^2)
ℂ^1
```
"""
function trivial(V::GradedSpace{I, D}) where {I, D}
 
     dims = TensorKit.SortedVectorDict(one(I) => 1)
     return GradedSpace{I,D}(dims, false)
end

function trivial(V::CartesianSpace)
     return ℝ^1
end

function trivial(V::ComplexSpace)
     return ℂ^1
end


"""
     istrivial(V::VectorSpace) -> ::Bool

Check if a given `VectorSpace` is trivial.

# Examples
```julia
julia> istrivial(Rep[U₁ × SU₂]((1, 1/2) => 2))
false

julia> istrivial(Rep[U₁ × SU₂]((0, 0) => 2))
false

julia> istrivial(Rep[U₁ × SU₂]((0, 0) => 1))
true
```
"""
function istrivial(V::VectorSpace)
     return V == trivial(V)
end
