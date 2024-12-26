using BenchmarkFreeFermions
using TensorKit
import LinearAlgebra.Diagonal

L = 8 # L must be even here
D = 512 # ≥ d^(L/2)
tol = 1e-8
duplicated = true

Hup = 2     # spin-↑ dop(+ hole; - electron)
Hdn = -1    # spin-↓ dop(+ hole; - electron)

# generate a random TB model
Tij = rand(ComplexF64, L, L)
Tij += Tij'

# exact results
ϵ, V = EigenModes(Tij)



# =============== DMRG ===============
Root = InteractionTreeNode()
for i in 1:L, j in 1:L
	if i == j
		addIntr1!(Root, U1SU2Fermion.n, i, -Tij[i, i]; name = :n)
	else
		addIntr2!(Root, U1SU2Fermion.FdagF, (i, j), -Tij[i, j]; name = (:Fdag, :F), Z = U1SU2Fermion.Z)
	end
end
H = AutomataMPO(InteractionTree(Root))

aspace = vcat(Rep[U₁×SU₂]((0, 0) => 1), repeat([Rep[U₁×SU₂]((i, j) => 1 for i in -1:1 for j in 0:1//2:1),], L-1)) # half-filling and S=0
Ψ = randMPS(ComplexF64, U1SU2Fermion.pspace, aspace)

Env = Environment(Ψ', H, Ψ)
lsEg = zeros(20)
for i in eachindex(lsEg)
	info, _ = DMRGSweep2!(Env; trunc = truncdim(D))
	lsEg[i] = info[2][1].Eg
end
Eg = lsEg[end]
errEg = Eg - 2*sum(ϵ[1:div(L, 2)])
@test abs(errEg) < tol # make sure DMRG works



# =============== calculate observables ===============
Tree = ObservableTree()
# 4-site
for i in 1:L, j in 1:L, k in 1:L, l in 1:L
	!duplicated && !allunique([i, j, k, l]) && continue
	addObs!(Tree, (U1SU2Fermion.FdagF..., U1SU2Fermion.FdagF...), (i, j, k, l), (true, true, true, true); Z = U1SU2Fermion.Z, name = (:Fdag, :F, :Fdag, :F))
	addObs!(Tree, Tuple(fill(U1SU2Fermion.n, 4)), (i, j, k, l), (false, false, false, false); name = (:n, :n, :n, :n))
end
# 3-site
for i in 1:L, j in 1:L, k in 1:L
	!duplicated && !allunique([i, j, k]) && continue
	addObs!(Tree, (U1SU2Fermion.n, U1SU2Fermion.n, U1SU2Fermion.n), (i, j, k), (false, false, false); name = (:n, :n, :n))
	addObs!(Tree, (U1SU2Fermion.FdagF..., U1SU2Fermion.n), (i, j, k), (true, true, false); Z = U1SU2Fermion.Z, name = (:Fdag, :F, :n))
	addObs!(Tree, U1SU2Fermion.SSS, (i, j, k), (false, false, false); name = (:S, :S, :S))
end
# 2-site
for i in 1:L, j in 1:L
	!duplicated && i == j && continue
	addObs!(Tree, U1SU2Fermion.FFdag, (i, j), (true, true);
		Z = U1SU2Fermion.Z, name = (:F, :Fdag))
	addObs!(Tree, U1SU2Fermion.FdagF, (i, j), (true, true);
		Z = U1SU2Fermion.Z, name = (:Fdag, :F))
	addObs!(Tree, (U1SU2Fermion.n, U1SU2Fermion.n), (i, j), (false, false); name = (:n, :n))
	addObs!(Tree, U1SU2Fermion.SS, (i, j), (false, false); name = (:S, :S))
end
# 1-site
for i in 1:L
     addObs!(Tree, U1SU2Fermion.n, i; name = :n)
end

calObs!(Tree, Ψ)
Obs = convert(NamedTuple, Tree)


# ================ operators ==================
# single partical Green function: G_{ij} = ⟨ϕL|c†_i c_j|ϕR⟩ / ⟨ϕL|ϕR⟩ = [ϕR * (ϕL' * ϕR)^{-1} * ϕL']_{ji}
function singreen(φL::AbstractVecOrMat{T1}, φR::AbstractVecOrMat{T2}) where {T1, T2}
    ((N,M) = size(φL)) == size(φR) || throw(DimensionMismatch("Mismatched Slater determinant dimensions: $(size(φL)) and $(size(φR))"))
    try
        return transpose(φR * inv(φL' * φR) * φL')
    catch e
        det(φL' * φR) == 0 && throw(ErrorException("Quadrature of Slater determinants: ⟨φ|φ′⟩=0"))
        rethrow(e)
    end
end
# const term (from e.g. c_i c†_j = δ_{ij} - c†_j c_i)
function expectation(::AbstractVecOrMat{T1}, O::Number, ::AbstractVecOrMat{T2}) where {T1, T2}
    return O
end
# expectation of monomer operator O = O_{ij} c†_i c_j
function expectation(φ::AbstractVecOrMat{T1}, O::AbstractMatrix{T2}, φ′::AbstractVecOrMat{T3}) where {T1, T2, T3}
    ((N,M) = size(φ)) == size(φ′) || throw(DimensionMismatch("Mismatched Slater determinant dimensions: $(size(φ)) and $(size(φ′))"))
    size(O) == (N, N) || throw(DimensionMismatch("Mismatched operator dimensions: $(size(O)) and ($N, $N)"))
    indices = findall(!iszero, O)
    G = singreen(φ, φ′)
    return sum(map(x -> O[x] * G[x], indices))
end
# expectation of two-body operator O = O_{ijkl} c†_i c†_j c_k c_l
function expectation(φ::AbstractVecOrMat{T1}, O::AbstractArray{T2, 4}, φ′::AbstractVecOrMat{T3}) where {T1, T2, T3}
    function _wick(index::CartesianIndex{4})
        i = index[1]; j = index[2]; k = index[3]; l = index[4];
        G = singreen(φ, φ′)
        return O[index] * (G[j,k] * G[i,l] - G[i,k] * G[j,l])
    end
    ((N,M) = size(φ)) == size(φ′) || throw(DimensionMismatch("Mismatched Slater determinant dimensions: $(size(φ)) and $(size(φ′))"))
    size(O) == (N, N, N, N) || throw(DimensionMismatch("Mismatched operator dimensions: $(size(O)) and ($N, $N, $N, $N)"))
    indices = findall(!iszero, O)
    return sum(map(x->_wick(x), indices))
end
# expectation of three-body operator O = O_{ijklmn} c†_i c†_j c†_k c_l c_m c_n
function expectation(φ::AbstractVecOrMat{T1}, O::AbstractArray{T2, 6}, φ′::AbstractVecOrMat{T3}) where {T1, T2, T3}
    function _wick(index::CartesianIndex{6})
        i = index[1]; j = index[2]; k = index[3]; l = index[4]; m = index[5]; n = index[6];
        G = singreen(φ, φ′)
        return O[index] * (G[k,l] * G[j,m] * G[i,n] + G[k,m] * G[j,n] * G[i,l] + G[k,n] * G[j,l] * G[i,m]
                         - G[k,l] * G[j,n] * G[i,m] - G[k,m] * G[j,l] * G[i,n] - G[k,n] * G[j,m] * G[i,l])
    end
    ((N,M) = size(φ)) == size(φ′) || throw(DimensionMismatch("Mismatched Slater determinant dimensions: $(size(φ)) and $(size(φ′))"))
    size(O) == (N, N, N, N, N, N) || throw(DimensionMismatch("Mismatched operator dimensions: $(size(O)) and ($N, $N, $N, $N, $N, $N)"))
    indices = findall(!iszero, O)
    return sum(map(x->_wick(x), indices))
end
# generic interface of `expectation`, O = O1_{ij} + O2_{ijkl} + O3_{ijklmn} + ⋯ --> (O1, O2, O3, ⋯)
function expectation(φ::AbstractVecOrMat{T1}, O::NTuple{N, Union{Number, AbstractArray{T2}}}, φ′::AbstractVecOrMat{T3}) where {T1, T2, T3, N}
    return sum(map(op->expectation(φ, op, φ′), O))
end


function PauliMatrix()
    σx = zeros(2, 2)
    σx[1,2] = σx[2,1] = 1

    σy = zeros(ComplexF64, 2, 2)
    σy[1,2] = -im; σy[2,1] = im

    σz = zeros(2, 2)
    σz[1,1] = 1; σz[2,2] = -1

    return σx, σy, σz
end


# operators
function op_n(i::Int)
    op1 = zeros(2*L, 2*L)
    op1[i, i] += 1
    op1[i+L, i+L] += 1
    return op1
end
function op_nn(i::Int, j::Int)
    op2 = zeros(2*L, 2*L, 2*L, 2*L)
    op2[i,j,i,j] -=1
    op2[i,j+L,i,j+L] -= 1
    op2[i+L,j,i+L,j] -= 1
    op2[i+L,j+L,i+L,j+L] -= 1

    if i != j
        return op2
    else
        return (op_n(i), op2)
    end
end
function op_SS(i::Int, j::Int)
    σx, σy, σz = PauliMatrix()

    op1 = zeros(2*L, 2*L)
    op2 = zeros(2*L, 2*L, 2*L, 2*L)
    for σ1=1:2, σ2=1:2, σ3=1:2, σ4=1:2
        value = 1/4 * (σx[σ1, σ2] * σx[σ3, σ4] + σy[σ1, σ2] * σy[σ3, σ4] + σz[σ1, σ2] * σz[σ3, σ4])
        if i == j && σ2 == σ3
            op1[i+(σ1-1)*L, j+(σ4-1)*L] += value
        end
        op2[i+(σ1-1)*L, j+(σ3-1)*L, i+(σ2-1)*L, j+(σ4-1)*L] -= value
    end

    return (op1, op2)
end
function op_FdagF(i::Int, j::Int)
    op1 = zeros(2*L, 2*L)
    op1[i, j] += 1
    op1[i+L, j+L] += 1
    return op1
end
function op_FFdag(i::Int, j::Int)
    op1 = zeros(2*L, 2*L)
    op1[j, i] -= 1
    op1[j+L, i+L] -= 1
    if i != j
        return op1
    else
        return (2, op1)
    end
end


# ================ test ==================
# exact results
ψ_ex = zeros(ComplexF64, 2*L, L)
ψ_ex[1:L, 1:div(L,2)] = V[:, 1:div(L,2)]            # ψup
ψ_ex[L+1:2*L, div(L,2)+1:L] = V[:, 1:div(L,2)]      # ψdn
G′ = singreen(ψ_ex, ψ_ex)

@testset "Boson-1" begin
	for i in 1:L
		@test haskey(Obs.n, (i,)) && abs(Obs.n[(i,)] - expectation(ψ_ex, op_n(i), ψ_ex)) < tol
	end
end

@testset "Boson-2" begin
	for i in 1:L, j in 1:L
		!duplicated && i == j && continue
		@test haskey(Obs.nn, (i, j)) && abs(Obs.nn[(i, j)] - expectation(ψ_ex, op_nn(i, j), ψ_ex)) < tol
		@test haskey(Obs.SS, (i, j)) && abs(Obs.SS[(i, j)] - expectation(ψ_ex, op_SS(i, j), ψ_ex)) < tol
	end
end

@testset "Fermion-2" begin
	for i in 1:L, j in 1:L
		!duplicated && i == j && continue
		@test haskey(Obs.FdagF, (i, j)) && abs(Obs.FdagF[(i, j)] - expectation(ψ_ex, op_FdagF(i, j), ψ_ex)) < tol
		@test haskey(Obs.FFdag, (i, j)) && real(Obs.FFdag[(i, j)] - expectation(ψ_ex, op_FFdag(i, j), ψ_ex)) < tol
	end
end

# @testset "Boson-3" begin
# 	for i in 1:L, j in 1:L, k in 1:L
# 		!duplicated && !allunique([i, j, k]) && continue
# 		@test haskey(Obs.nnn, (i,j,k)) && abs(Obs.nnn[(i, j, k)] - expectation(ψ_ex, op_nnn(i, j), ψ_ex)) < tol
# 	end
# end

# @testset "Boson-1-Fermion-2" begin
# 	for i in 1:L, j in 1:L, k in 1:L
# 		!duplicated && !allunique([i, j, k]) && continue
# 		@test haskey(Obs.FdagFn, (i, j, k)) && abs(Obs.FdagFn[(i, j, k)] - expectation(ψ_ex, op_FdagFn(i, j), ψ_ex)) < tol
# 	end
# end

# @testset "Boson-4" begin
# 	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
# 		!duplicated && !allunique([i, j, k, l]) && continue
# 		@test haskey(Obs.nnnn, (i, j, k, l)) && abs(Obs.nnnn[(i, j, k, l)] - expectation(ψ_ex, op_FdagFn(i, j), ψ_ex)) < tol
# 	end
# end

# @testset "Fermion-4" begin
# 	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
# 		!duplicated && !allunique([i, j, k, l]) && continue
# 		@test haskey(Obs.FdagFFdagF, (i, j, k, l)) && abs(Obs.FdagFFdagF[(i, j, k, l)] - expectation(ψ_ex, op_FdagFn(i, j), ψ_ex)) < tol
# 	end
# end

