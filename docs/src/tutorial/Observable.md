# Observable

We use a similar tree structure to calculating observables. For instance, we first generate a random product state. 
```@setup Observable
using FiniteMPS
```
```@example Observable
# construct a random MPS 
L = 4
Ψ = randMPS(L, ℂ^2, ℂ^1)
# display
for i in 1:L
println(Ψ[i])
end
```
Here we use `randMPS` to construct a random MPS whose physical spaces are all $\mathbb{C}^2$ and bond spaces are all $\mathbb{C}^1$ thus represents a random produce state. 

Next, we calculate the on-site values and spin correlations.
```@example Observable
# define the S^z operator following the syntax of TensorKit.jl
Sz = TensorMap([0.5 0.0; 0.0 -0.5], ℂ^2, ℂ^2)

# construct the observable tree
Tree = ObservableTree(L)
for i in 1:L
     addObs!(Tree, (Sz,), (i,), (false,); name = (:Sz,))
end
for i in 1:L, j in 1:L
     addObs!(Tree, (Sz, Sz), (i, j), (false, false); name = (:Sz, :Sz))
end
calObs!(Tree, Ψ)
Obs = convert(Dict, Tree)
```
Here `Tree` is an `ObservableTree` object that contains all observables to be calculated, and `addObs!` is the standard interface to add terms to it, analog to `InteractionTree` and `addIntr!`. Then, we call `calObs!` to trigger the in-place calculation in the tree, with the given MPS `Ψ`. Finally, we use the `convert` method to extract the data from the tree to a dictionary `Obs`. For example, 
```@example Observable
Obs["SzSz"][(1, 2)]
```
is the correlation `\langle S_1^z S_2^z\rangle`. One can perform a simple quantum mechanics calculation to check this result.

Here is just a simple example to show the basic usage, more complex examples that contain fermion correlations and multi-site correlations (e.g. pairing correlations) can be found in the concrete example for [Hubbard model](@ref Hubbard).
