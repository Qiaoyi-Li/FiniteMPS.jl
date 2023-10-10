module FiniteMPS

using TimerOutputs
using Reexport
using AbstractTrees, SerializedElementArrays
using Base.Threads, FLoops, FoldsThreads, Distributed
@reexport import SerializedElementArrays: SerializedElementArray,  SerializedElementVector
@reexport using TensorKit, KrylovKit, TensorKit.TensorOperations
@reexport import Base: +, -, *, /, ==, promote_rule, convert, length, show, getindex, setindex!, lastindex, keys, similar
@reexport import TensorKit: ×, one, zero, dim, inner, scalar, domain, codomain, eltype, scalartype, leftorth, rightorth, tsvd, adjoint, normalize!, norm, axpy!, axpby!, dot, mul!, rmul!
@reexport import LinearAlgebra: rank, qr

# Default settings
include("Defaults.jl")

# Utils
export trivial, istrivial, data, UniformDistribution, GaussianDistribution, NormalDistribution, randStiefel, randisometry, randisometry!
include("utils/trivial.jl")
include("utils/TensorMap.jl")
include("utils/Random.jl")
include("utils/CompatThreading.jl")
include("utils/SerializedElementArrays.jl")

# Wrapper types for classifying tensors
export AbstractTensorWrapper, AbstractMPSTensor, MPSTensor, CompositeMPSTensor, AdjointMPSTensor, AbstractEnvironmentTensor, LocalLeftTensor, LocalRightTensor, SimpleLeftTensor, SimpleRightTensor, SparseLeftTensor, SparseRightTensor, AbstractLocalOperator, hastag, getOpName, IdentityOperator, tag2Tuple, LocalOperator, SparseMPOTensor, AbstractStoreType, StoreMemory, StoreDisk
include("TensorWrapper/TensorWrapper.jl")
include("TensorWrapper/MPSTensor.jl")
include("TensorWrapper/CompositeMPSTensor.jl")
include("TensorWrapper/AdjointMPSTensor.jl")
include("TensorWrapper/EnvironmentTensor.jl")
include("TensorWrapper/LocalOperator.jl")
include("TensorWrapper/SparseMPOTensor.jl")
include("TensorWrapper/StoreType.jl")

# Dense MPS and MPO
export AbstractMPS, DenseMPS, AdjointMPS, coef, Center, MPS, randMPS, canonicalize!
include("MPS/AbstractMPS.jl")
include("MPS/AdjointMPS.jl")
include("MPS/MPS.jl")
include("MPS/canonicalize.jl")
# TODO MPO

# Sparse MPO
export SparseMPO, issparse
include("SparseMPO/SparseMPO.jl")

# Environment
export AbstractEnvironment, SimpleEnvironment, SparseEnvironment, Environment, pushleft!, pushright!, canonicalize!, free!, scalar!
include("Environment/Environment.jl")
include("Environment/initialize.jl")
include("Environment/pushleft.jl")
include("Environment/pushright.jl")
include("Environment/canonicalize.jl")
include("Environment/scalar.jl")

# Projective Hamiltonian
export AbstractProjectiveHamiltonian, SparseProjectiveHamiltonian, ProjHam, action2
include("ProjectiveHam/ProjectiveHam.jl")
include("ProjectiveHam/action2.jl")
# TODO action1

# Algorithm
export LanczosInfo, DMRG2Info, DMRG1Info,  DMRGSweep2!
include("Algorithm/DMRG.jl")

# Interaction tree for generating Hamiltonian MPO and calculate observables
export InteractionTreeNode, InteractionTree, addchild!, addIntr!, addIntr1!, addIntr2!, addIntr4!, AutomataMPO
include("IntrTree/Node.jl")
include("IntrTree/addIntr.jl")
include("IntrTree/addIntr1.jl")
include("IntrTree/addIntr2.jl")
include("IntrTree/addIntr4.jl")
include("IntrTree/Automata.jl")


# Observables
export calObs!, ObservableTree, addObs!
include("Observables/ObsTree.jl")
include("Observables/calObs.jl")
include("Observables/convert.jl")


# predefined local spaces
export U₁SU₂Fermion, U1SU2Fermion
include("LocalSpace/Fermion.jl")


end # module FiniteMPS
