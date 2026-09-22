# Imaginary Time Proxy

For computing imaginary time proxies (ITP), paired local operators share prefix and suffix trees. `addITP!` accepts two nonempty operator tuples and flat tuples of sites, fermionic flags and names. Each operator chain is sorted and reduced independently; result keys keep the original site order. `calITP!` uses the same `LayeredTreeEval` execution and environment cache as `calObs!`.

```@docs
ImagTimeProxyTree
merge!(::ImagTimeProxyTree)
addITP!
calITP!
convert
```

