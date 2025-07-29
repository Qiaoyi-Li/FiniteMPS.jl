# Algorithms

Some high level algorithms based on MPS and MPO.

## DMRG

Density matrix renormalization group.

```@docs
BondInfo
DMRGInfo
LanczosGS
DMRGSweep2!
DMRGSweep1!
```

## TDVP

Time-dependent variational principle for both MPS and MPO (viewed as a MPS via Choi isomorphism).

Note the local projective Hamiltonian action in `TDVPSweep2!` has not been updated to newest implementation. Using 1-TDVP equipped with controlled bond expansion (CBE) is preferred. 

```@docs
TDVPInfo
LanczosExp
TDVPSweep2!
TDVPSweep1!
TDVPIntegrator
SymmetricIntegrator
```

## SETTN

Series-expansion thermal tensor network.

```@docs
SETTN
```

## CBE

Controlled bond expansion (CBE). In the current version only a modified (based on rsvd, instead of shrewd selection) CBE is implemented, which is compatible with multi-threading.

```@docs
CBEAlgorithm
NoCBE
FullCBE
NaiveCBE
CBEInfo
CBE
LeftOrthComplement
RightOrthComplement
```






