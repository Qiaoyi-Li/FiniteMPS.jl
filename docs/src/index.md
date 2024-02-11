# FiniteMPS.jl

```@meta
CurrentModule = FiniteMPS
```

## Contents
```@contents
```

## Types
### Tensor wrappers
```@docs
AbstractTensorWrapper
AbstractMPSTensor
MPSTensor
CompositeMPSTensor
AdjointMPSTensor
AbstractEnvironmentTensor 
LocalLeftTensor
LocalRightTensor
SimpleLeftTensor
SimpleRightTensor
SparseLeftTensor
SparseRightTensor
AbstractLocalOperator
IdentityOperator
tag2Tuple
LocalOperator
SparseMPOTensor
AbstractStoreType
StoreMemory
StoreDisk
```

### MPS/MPO
```@docs
AbstractMPS
DenseMPS
AdjointMPS
MPS
MPO
SparseMPO
```

### Environment
```@docs
AbstractEnvironment
SimpleEnvironment
SparseEnvironment
Environment
```

### Projective Hamiltonian
```@docs
AbstractProjectiveHamiltonian
IdentityProjectiveHamiltonian
SparseProjectiveHamiltonian
```

### Interaction tree
```@docs
InteractionTreeNode
InteractionTree
addIntr!
addIntr1!
addIntr2!
addIntr4!
AutomataMPO
ObservableTree
addObs!
```

## Algorithms
### DMRG
```@docs
DMRGInfo
DMRGSweep2!
DMRGSweep1!
```

### TDVP
```@docs
TDVPInfo
TDVPSweep2!
TDVPSweep1!
TDVPIntegrator
SymmetricIntegrator
```

### CBE
```@docs
CBEAlgorithm
NoCBE
FullCBE
StandardCBE
CBE
```
