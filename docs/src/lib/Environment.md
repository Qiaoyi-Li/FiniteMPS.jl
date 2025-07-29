# Environment

Deal with the left and right environment tensors of multi-layer MPS/MPO structure. 

```@docs
AbstractEnvironment
Center(::AbstractEnvironment)
SimpleEnvironment
SparseEnvironment
Environment
initialize!
pushleft!
pushright!
canonicalize!(::AbstractEnvironment)
free!(::AbstractEnvironment)
scalar!
connection!
absorb!
```