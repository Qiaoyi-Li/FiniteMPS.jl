# Observable Tree

Similar to `InteractionTree`, for computing expectation values of observables. Both `calObs!` and `calITP!` accept `alg=LayeredTreeEval(ntasks=N)`, with all default-pool Julia threads used by default; `N=1` runs synchronously. In disk mode, `maxsize` limits the total number of cached environments across both trees. Its default is the widest combined layer, and zero disables caching. `GCspacing=0` disables periodic manual collection.

```@docs
ObservableTree
merge!(::ObservableTree)
treewidth
addObs!
calObs!
```