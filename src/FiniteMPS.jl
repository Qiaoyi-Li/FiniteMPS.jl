module FiniteMPS

using Reexport
using AbstractTrees, SerializedElementArrays, Serialization
using Base.Threads, Distributed
using Graphs, MetaGraphs
@reexport import SerializedElementArrays: SerializedElementArray, SerializedElementVector
@reexport using TensorKit, KrylovKit, TensorKit.TensorOperations, TimerOutputs
@reexport import Base: +, -, *, /, ==, promote_rule, convert, length, show, getindex, setindex!, lastindex, keys, similar, merge, merge!, iterate, complex
@reexport import TensorKit: ×, one, zero, dim, inner, scalar, domain, codomain, eltype, scalartype, leftorth, rightorth, leftnull, rightnull, tsvd, adjoint, normalize!, norm, axpy!, axpby!, add!, add!!, dot, mul!, rmul!, NoTruncation, fuse, zerovector!, zerovector, scale, scale!, scale!!
@reexport import LinearAlgebra: BLAS, rank, qr, diag, I, diagm
import AbstractTrees: parent, isroot
import Graphs: rem_vertices!

# global settings
include("Globals.jl")
include("Defaults.jl")

# Utils
export trivial, istrivial, data, UniformDistribution, GaussianDistribution, NormalDistribution, randStiefel, randisometry, randisometry!, cleanup!, oplusEmbed, SweepDirection, SweepL2R, SweepR2L, AnyDirection
include("utils/trivial.jl")
include("utils/TensorMap.jl")
include("utils/Random.jl")
include("utils/CompatThreading.jl")
include("utils/SerializedElementArrays.jl")
include("utils/manualGC.jl")
include("utils/cleanup.jl")
include("utils/oplus.jl")
include("utils/Direction.jl")

# Wrapper types for classifying tensors
export AbstractTensorWrapper, AbstractMPSTensor, MPSTensor, CompositeMPSTensor, AdjointMPSTensor, AbstractEnvironmentTensor, LocalLeftTensor, LocalRightTensor, SimpleLeftTensor, SimpleRightTensor, SparseLeftTensor, SparseRightTensor, BilayerLeftTensor, BilayerRightTensor, AbstractLocalOperator, hastag, getPhysSpace, getOpName, isfermionic, IdentityOperator, tag2Tuple, LocalOperator, StringOperator, SparseMPOTensor, AbstractStoreType, StoreMemory, StoreDisk, LeftPreFuseTensor, SparseLeftPreFuseTensor, noise!
include("TensorWrapper/TensorWrapper.jl")
include("TensorWrapper/MPSTensor.jl")
include("TensorWrapper/CompositeMPSTensor.jl")
include("TensorWrapper/AdjointMPSTensor.jl")
include("TensorWrapper/EnvironmentTensor.jl")
include("TensorWrapper/BilayerEnvTensor.jl")
include("TensorWrapper/LocalOperator.jl")
include("TensorWrapper/StringOperator.jl")
include("TensorWrapper/SparseMPOTensor.jl")
include("TensorWrapper/StoreType.jl")
include("TensorWrapper/PreFuseTensor.jl")

# Dense MPS
export AbstractMPS, DenseMPS, AdjointMPS, coef, Center, MPS, randMPS, canonicalize!
include("MPS/AbstractMPS.jl")
include("MPS/AdjointMPS.jl")
include("MPS/MPS.jl")
include("MPS/canonicalize.jl")

# Dense MPO
export MPO, identityMPO
include("MPO/MPO.jl")

# Sparse MPO
export SparseMPO, issparse
include("SparseMPO/SparseMPO.jl")

# Environment
export AbstractEnvironment, SimpleEnvironment, SparseEnvironment, Environment, initialize!, pushleft!, pushright!, canonicalize!, free!, scalar!, connection!
include("Environment/Environment.jl")
include("Environment/initialize.jl")
include("Environment/pushleft.jl")
include("Environment/pushright.jl")
include("Environment/canonicalize.jl")
include("Environment/scalar.jl")
include("Environment/connection.jl")

# Projective Hamiltonian
export AbstractProjectiveHamiltonian, IdentityProjectiveHamiltonian, SparseProjectiveHamiltonian, ProjHam, action2, action1, action0, PreFuseProjectiveHamiltonian
include("ProjectiveHam/ProjectiveHam.jl")
include("ProjectiveHam/prefuse.jl")
include("ProjectiveHam/action2.jl")
include("ProjectiveHam/action1.jl")
include("ProjectiveHam/action0.jl")

# Algebra operations
include("Algebra/inner.jl")
include("Algebra/mul.jl")
include("Algebra/axpby.jl")

# Algorithm
export LanczosInfo, BondInfo, DMRGInfo, TDVPInfo, DMRGSweep2!, DMRGSweep1!, SETTN, TDVPSweep2!, TDVPSweep1!, TDVPIntegrator, SymmetricIntegrator, CBEAlgorithm, NoCBE, FullCBE, StandardCBE, CheapCBE, CBE, LeftOrthComplement, RightOrthComplement
include("Algorithm/Info.jl")
include("Algorithm/DMRG.jl")
include("Algorithm/SETTN.jl")
include("Algorithm/TDVP/TDVPUpdate.jl")
include("Algorithm/TDVP/TDVP2.jl")
include("Algorithm/TDVP/TDVP1.jl")
include("Algorithm/TDVP/Integrator.jl")
include("Algorithm/CBE/utils.jl")
include("Algorithm/CBE/OrthComplement.jl")
include("Algorithm/CBE/preselect.jl")
include("Algorithm/CBE/finalselect.jl")
include("Algorithm/CBE/SparseSVD.jl")
include("Algorithm/CBE/CBEAlgorithm.jl")
include("Algorithm/CBE/CBE.jl")


# Interaction tree for generating Hamiltonian MPO and calculate observables
export InteractionTreeNode, InteractionTree, addchild!, addIntr!, addIntr1!, addIntr2!, addIntr4!, AutomataMPO, AbstractInteractionIterator, OnSiteInteractionIterator, TwoSiteInteractionIterator, ArbitraryInteractionIterator
include("IntrTree/Node.jl")
include("IntrTree/addIntr.jl")
include("IntrTree/addIntr1.jl")
include("IntrTree/addIntr2.jl")
include("IntrTree/addIntr4.jl")
include("IntrTree/Automata.jl")
include("IntrTree/IntrIterator.jl")


# Observables
export calObs!, ObservableTree, addObs!, ImagTimeProxyGraph, addITP2!, addITP4!, calITP!
include("Observables/ObsTree.jl")
include("Observables/calObs.jl")
include("Observables/convert.jl")
include("Observables/ITPGraph.jl")
include("Observables/addITP.jl")
include("Observables/calITP.jl")
include("Observables/pushleft.jl")
include("Observables/pushright.jl")

# predefined local spaces
export SU₂Spin, SU2Spin, U₁Spin, U1Spin, NoSymSpinOneHalf, U₁SU₂Fermion, U1SU2Fermion, ℤ₂SU₂Fermion, Z2SU2Fermion, U₁SpinlessFermion, U1SpinlessFermion, U₁SU₂tJFermion, U1SU2tJFermion, U₁U₁Fermion, U1U1Fermion, U₁U₁tJFermion, U1U1tJFermion, ℤ₂SU₂tJFermion, Z2SU2tJFermion
include("LocalSpace/Spin.jl")
include("LocalSpace/Fermion.jl")
include("LocalSpace/tJFermion.jl")
include("LocalSpace/SpinlessFermion.jl")


# __init__
include("init.jl")

end # module FiniteMPS
