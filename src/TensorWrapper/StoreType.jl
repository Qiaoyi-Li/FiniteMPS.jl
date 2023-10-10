"""
     abstract type AbstractStoreType

Only has 2 concrete types `StoreMemory` and `StoreDisk`, to determine how the collections such as `MPS`, `MPO` and  `Environment` store the local tensors.
"""
abstract type AbstractStoreType end

"""
     struct StoreMemory <: AbstractStoreType

Tell the collection to store local tensors in memory.
"""
struct StoreMemory <: AbstractStoreType end

"""
     struct StoreDisk <: AbstractStoreType

Tell the collection to store local tensors in disk.
"""
struct StoreDisk <: AbstractStoreType end