using BenchmarkFreeFermions
import LinearAlgebra.Diagonal, LinearAlgebra.det
using Combinatorics

L = 6 # L must be even here
D = 64 # ≥ d^(L/2)
tol = 1e-8
tol4 = 1e-8
duplicated = true

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

function wick(G::Matrix, i::Int, j::Int, k::Int, l::Int)
    return G[j,k] * G[i,l] - G[i,k] * G[j,l]
end
function wick(G::Matrix, i::Int, j::Int, k::Int, l::Int, m::Int, n::Int)
    return G[k,l] * G[j,m] * G[i,n] + G[k,m] * G[j,n] * G[i,l] + G[k,n] * G[j,l] * G[i,m] - G[k,l] * G[j,n] * G[i,m] - G[k,m] * G[j,l] * G[i,n] - G[k,n] * G[j,m] * G[i,l]
end
function wick(G::Matrix, i::Int, j::Int, k::Int, l::Int, m::Int, n::Int, p::Int, q::Int)
    index_index_permutation = collect(permutations(1:4))
    index = (m, n, p, q)
    return sum((_permutation_sign(x)*G[l,index[x[1]]]*G[k,index[x[2]]]*G[j,index[x[3]]]*G[i,index[x[4]]] for x in index_index_permutation))
end
function _permutation_sign(p)
    n = length(p)
    @assert Set(p) == Set(1:n)
    perm_matrix = zeros(Int, n, n)
    for (i, val) in enumerate(p)
        perm_matrix[i, val] = 1
    end
    return det(perm_matrix) == 1 ? 1 : -1
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
function ex_n(G::Matrix, i::Int)
    ex = 0
    for σ1=1:2
        i1 = i + (σ1-1)*L

        ex += G[i1,i1]
    end
    return ex
end
function ex_nn(G::Matrix, i::Int, j::Int)
    ex = 0
    for σ1=1:2, σ2=1:2
        i1 = i+(σ1-1)*L; j2 = j+(σ2-1)*L;

        ex -= wick(G, i1, j2, i1, j2)

        if (i==j) && (σ1==σ2)
            ex += G[i1, j2]
        end
    end
    return ex
end
function ex_SS(G::Matrix, i::Int, j::Int)
    σx, σy, σz = PauliMatrix()
    ex = 0
    for σ1=1:2, σ2=1:2, σ3=1:2, σ4=1:2
        value = 1/4 * (σx[σ1, σ2] * σx[σ3, σ4] + σy[σ1, σ2] * σy[σ3, σ4] + σz[σ1, σ2] * σz[σ3, σ4])
        i1 = i+(σ1-1)*L; j3 = j+(σ3-1)*L; i2 = i+(σ2-1)*L; j4 = j+(σ4-1)*L;

        ex -= value * wick(G, i1,j3,i2,j4)

        if (i==j) && (σ2==σ3)
            ex += value * G[i1, j4]
        end
    end
    return ex
end
function ex_FdagF(G::Matrix, i::Int, j::Int)
    ex = 0
    for σ1=1:2
        i1 = i+(σ1-1)*L; j1 = j+(σ1-1)*L;

        ex += G[i1, j1]
    end
    return ex
end
function ex_FFdag(G::Matrix, i::Int, j::Int)
    ex = 0
    for σ1=1:2
        i1 = i+(σ1-1)*L; j1 = j+(σ1-1)*L;

        ex -= G[j1, i1]

        if (i==j)
            ex += 1
        end
    end
    return ex
end
function ex_nnn(G::Matrix, i::Int, j::Int, k::Int)
    ex = 0
    for σ1=1:2, σ2=1:2, σ3=1:2
        i1 = i + (σ1-1)*L; j2 = j+(σ2-1)*L; k3 = k+(σ3-1)*L;

        ex -= wick(G, i1, j2, k3, i1, j2, k3)

        if (i==j) && (σ1==σ2)
            ex -= wick(G, i1, k3, j2, k3)
        end
        if (j==k) && (σ2==σ3)
            ex -= wick(G, i1,j2,i1,k3)
        end
        if (i==k) && (σ1==σ3)
            ex += wick(G, i1, j2, j2, k3)
        end

        if (i==j==k) && (σ1==σ2==σ3)
            ex += G[i1,k3]
        end
    end
    return ex
end
function ex_SSS(G::Matrix, i::Int, j::Int, k::Int)
    σx, σy, σz = PauliMatrix()
    ex = 0
    for σ1=1:2, σ2=1:2, σ3=1:2, σ4=1:2, σ5=1:2, σ6=1:2
        value = 1/8 * (σx[σ1, σ2] * σy[σ3, σ4] * σz[σ5, σ6]
                     + σy[σ1, σ2] * σz[σ3, σ4] * σx[σ5, σ6]
                     + σz[σ1, σ2] * σx[σ3, σ4] * σy[σ5, σ6]
                     - σz[σ1, σ2] * σy[σ3, σ4] * σx[σ5, σ6]
                     - σy[σ1, σ2] * σx[σ3, σ4] * σz[σ5, σ6]
                     - σx[σ1, σ2] * σz[σ3, σ4] * σy[σ5, σ6])
        i1 = i+(σ1-1)*L; i2 = i+(σ2-1)*L; j3 = j+(σ3-1)*L; j4 = j+(σ4-1)*L; k5 = k+(σ5-1)*L; k6 = k+(σ6-1)*L;

        ex -= value * wick(G, i1, j3, k5, i2, j4, k6)

        if (σ2==σ3) && (i==j)
            ex -= value * wick(G, i1, k5, j4, k6)
        end
        if (σ4==σ5) && (j==k)
            ex -= value * wick(G, i1, j3, i2, k6)
        end
        if (σ2==σ5) && (i==k)
            ex += value * wick(G, i1, j3, j4, k6)
        end

        if (σ2==σ3) && (σ4==σ5) && (i==j==k)
            ex += value * G[i1, k6]
        end
    end
    return ex
end
function ex_FdagFn(G::Matrix, i::Int, j::Int, k::Int)
    ex = 0
    for σ1 = 1:2, σ2 = 1:2
        i1 = i+(σ1-1)*L; j1 = j+(σ1-1)*L; k2 = k+(σ2-1)*L;

        ex -= wick(G, i1, k2, j1, k2)

        if (σ1==σ2) && (j==k)
            ex += G[i1,k2]
        end
    end
    return ex
end
function ex_nnnn(G::Matrix, i::Int, j::Int, k::Int, l::Int)
    ex = 0
    for σ1 = 1:2, σ2 = 1:2, σ3 = 1:2, σ4 = 1:2
        i1 = i+(σ1-1)*L; j2 = j+(σ2-1)*L; k3 = k+(σ3-1)*L; l4 = l+(σ4-1)*L;

        ex += wick(G, i1, j2, k3, l4, i1, j2, k3, l4)

        if (i==j) && (σ1==σ2)
            ex -= wick(G, i1, k3, l4, j2, k3, l4)
        end
        if (j==k) && (σ2==σ3)
            ex -= wick(G, i1, j2, l4, i1, k3, l4)
        end
        if (k==l) && (σ3==σ4)
            ex -= wick(G, i1, j2, k3, i1, j2, l4)
        end
        if (l==i) && (σ4==σ1)
            ex -= wick(G, i1, j2, k3, j2, k3, l4)
        end
        if (i==k) && (σ1==σ3)
            ex += wick(G, i1, j2, l4, j2, k3, l4)
        end
        if (j==l) && (σ2==σ4)
            ex += wick(G, i1, j2, k3, i1, k3, l4)
        end

        if (i==j==k) && (σ1==σ2==σ3)
            ex -= wick(G, i1, l4, k3, l4)
        end
        if (i==j) && (k==l) && (σ1==σ2) && (σ3==σ4)
            ex -= wick(G, i1, k3, j2, l4)
        end
        if (j==k==l) && (σ2==σ3==σ4)
            ex -= wick(G, i1, j2, i1, l4)
        end
        if (i==k) && (j==l) && (σ1==σ3) && (σ2==σ4)
            ex -= wick(G, i1, j2, k3, l4)
        end
        if (i==j==l) && (σ1==σ2==σ4)
            ex += wick(G, i1, k3, k3, l4)
        end
        if (i==l) && (j==k) && (σ1==σ4) && (σ2==σ3)
            ex += wick(G, i1, j2, k3, l4)
        end
        if (i==k==l) && (σ1==σ3==σ4)
            ex += wick(G, i1, j2, j2, l4)
        end

        if (i==j==k==l) && (σ1==σ2==σ3==σ4)
            ex += G[i1, l4]
        end
    end
    return ex
end
function ex_FdagFFdagF(G::Matrix, i::Int, j::Int, k::Int, l::Int)
    ex = 0
    for σ1 = 1:2, σ2 = 1:2
        i1 = i+(σ1-1)*L; j1 = j+(σ1-1)*L; k2 = k+(σ2-1)*L; l2 = l+(σ2-1)*L

        ex -= wick(G, i1, k2, j1, l2)

        if (j==k) && (σ1==σ2)
            ex += G[i1,l2]
        end
    end
    return ex
end


# ================ test ==================
# exact results
ψ_ex = zeros(ComplexF64, 2*L, L)
ψ_ex[1:L, 1:div(L,2)] = V[:, 1:div(L,2)]            # ψup
ψ_ex[L+1:2*L, div(L,2)+1:L] = V[:, 1:div(L,2)]      # ψdn
G′ = Matrix(singreen(ψ_ex, ψ_ex))

@testset "Boson-1" begin
	for i in 1:L
		@test haskey(Obs.n, (i,)) && abs(Obs.n[(i,)] - ex_n(G′, i)) < tol
	end
end

@testset "Boson-2" begin
	for i in 1:L, j in 1:L
		!duplicated && i == j && continue
		@test haskey(Obs.nn, (i, j)) && abs(Obs.nn[(i, j)] - ex_nn(G′, i, j)) < tol
		@test haskey(Obs.SS, (i, j)) && abs(Obs.SS[(i, j)] - ex_SS(G′, i, j)) < tol
	end
end

@testset "Fermion-2" begin
	for i in 1:L, j in 1:L
		!duplicated && i == j && continue
		@test haskey(Obs.FdagF, (i, j)) && abs(Obs.FdagF[(i, j)] - ex_FdagF(G′, i, j)) < tol
		@test haskey(Obs.FFdag, (i, j)) && abs(Obs.FFdag[(i, j)] - ex_FFdag(G′, i, j)) < tol
	end
end

@testset "Boson-3" begin
	for i in 1:L, j in 1:L, k in 1:L
		!duplicated && !allunique([i, j, k]) && continue
		@test haskey(Obs.nnn, (i,j,k)) && abs(Obs.nnn[(i, j, k)] - ex_nnn(G′, i, j, k)) < tol
		@test haskey(Obs.SSS, (i,j,k)) && abs(im*Obs.SSS[(i, j, k)] - ex_SSS(G′, i, j, k)) < tol
        allunique([i, j, k]) && @test real(Obs.SSS[(i, j, k)]) < tol
	end
end

@testset "Boson-1-Fermion-2" begin
	for i in 1:L, j in 1:L, k in 1:L
		!duplicated && !allunique([i, j, k]) && continue
		@test haskey(Obs.FdagFn, (i, j, k)) && abs(Obs.FdagFn[(i, j, k)] - ex_FdagFn(G′, i, j, k)) < tol
	end
end

@testset "Boson-4" begin
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		!duplicated && !allunique([i, j, k, l]) && continue
		@test haskey(Obs.nnnn, (i, j, k, l)) && abs(Obs.nnnn[(i, j, k, l)] - ex_nnnn(G′, i, j, k, l)) < tol4
	end
end

@testset "Fermion-4" begin
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		duplicated && !allunique([i, j, k, l]) && continue
		@test haskey(Obs.FdagFFdagF, (i, j, k, l)) && abs(Obs.FdagFFdagF[(i, j, k, l)] - ex_FdagFFdagF(G′, i, j, k, l)) < tol4
	end
end
