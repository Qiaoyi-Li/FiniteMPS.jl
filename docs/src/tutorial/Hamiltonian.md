# Hamiltonian

Almost all algorithms in FiniteMPS.jl are based on a Hamiltonian MPO, so the starting point is always generating the Hamiltonian MPO of the given model. We provide a standard approach to this requirement.
```@example Hamiltonian
using FiniteMPS

# generate Hamiltonian of a length-L Ising chain
L = 4
J = 1.0
Tree = InteractionTree(L)
for i in 1:L-1
     addIntr!(Tree, (U₁Spin.Sz, U₁Spin.Sz), (i, i+1), (false, false), J; name = (:Sz, :Sz))
end
H = AutomataMPO(Tree)
```
Here, we generate the Hamiltonian of a length-4 Ising chain. We first construct an empty `InteractionTree` object `Tree`, which is used to store all interactions with a bi-tree structure. 

Then, we add the interaction terms one by one via a standard interface `addIntr!`, where a N-site interaction is characterized by three N-tuple of operators, sites, and fermion flags. For a Ising coupling, `N=2`. `U₁Spin.Sz` is the predefined spin operator with U(1) symmetry, more predefined operators are listed in [LocalSpace](@ref LocalSpace) section. `(i, i+1)` indicates the two operators are located in site `i` and `i+1`. `(false, false)` means both operators are bosonic.

Finally, we use `AutomataMPO` to construct the Hamiltonian represented by a sparse MPO, whose local tensor is an abstract matrix (gives 2 bond indices) of local rank-2 operators (give 2 physical indices).
```@example Hamiltonian
# show the local tensors
for i in 1:L
     println(H[i])
end
```
One can check the result via matrix multiplication, which corresponds to contracting the bond indices. For example, multiplying the first two matrices gives $[I_1S_2^z,\ I_1I_2,\ S_1^zS_2^z]$. Continue to multiply the third matrix and we obtain $[I_1S_2^zS_3^z + S_1^zS_2^zI_3,\ I_1I_2S_3^z]$. Finally, multiplying the last matrix indeed gives the total Ising Hamiltonian $H = S_1^zS_2^z + S_2^zS_3^z + S_3^zS_4^z$.


