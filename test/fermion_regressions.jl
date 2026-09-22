using FiniteMPS: ×

function _dense_mps(Ψ)
    state = ones(ComplexF64, 1, 1)
    for A in Ψ
        B = convert(Array, A.A)
        state = reshape(state * reshape(B, size(B, 1), :), :, size(B, 3))
    end
    return vec(state) * coef(Ψ)
end

function _dense_mpo(H)
    paths = Dict(1 => [ones(ComplexF64, 1, 1)])
    for M in H
        next = Dict{Int, Vector{Matrix{ComplexF64}}}()
        for j in axes(M, 2), i in axes(M, 1)
            O = M[i, j]
            isnothing(O) && continue
            left, right = dim(getLeftSpace(O)), dim(getRightSpace(O))
            d = dim(getPhysSpace(O))
            A = O isa IdentityOperator ? Matrix{Float64}(I, d, d) : convert(Array, O.A)
            blocks = get!(next, j) do
                [zeros(ComplexF64, size(paths[i][1], 1) * d, size(paths[i][1], 2) * d) for _ in 1:right]
            end
            for r in 1:right, l in 1:left
                B = if ndims(A) == 2
                    l == r || continue
                    A
                else
                    reshape(A, left, d, d, right)[l, :, :, r]
                end
                blocks[r] .+= O.strength[] .* kron(B, paths[i][l])
            end
        end
        paths = next
    end
    return paths[1][1]
end

function _fermion_modes(local_modes, L)
    d = size(first(local_modes), 1)
    parity = prod(Matrix{Float64}(I, d, d) - 2c' * c for c in local_modes)
    return [[foldr(kron, [j > i ? parity : j == i ? c : Matrix{Float64}(I, d, d) for j in L:-1:1]) for i in 1:L] for c in local_modes]
end

@testset "Fermion registration" begin
    spinless = zeros(2, 2)
    v = U1SpinlessFermion.pspace
    spinless[only(axes(v, U1Irrep(-1//2))), only(axes(v, U1Irrep(1//2)))] = 1
    v = U1U1Fermion.pspace
    empty, u, d, double = (only(axes(v, Irrep[U₁×U₁](q...))) for q in ((-1, 0), (0, 1//2), (0, -1//2), (1, 0)))
    up, down = zeros(4, 4), zeros(4, 4)
    up[empty, u] = up[d, double] = down[empty, d] = 1
    down[u, double] = -1
    L = 4
    hopping = ComplexF64[0.4 0.3+0.2im 0. 0.7-0.1im; 0.3-0.2im -0.2 0.5im 0.; 0. -0.5im 0.1 -0.6; 0.7+0.1im 0. -0.6 -0.3]
    @testset "$F" for (F, local_modes, FdagF, FFdag, densities) in (
        (U1SpinlessFermion, [spinless], (U1SpinlessFermion.FdagF,), (U1SpinlessFermion.FFdag,), (U1SpinlessFermion.n,)),
        (U1U1Fermion, [up, down], (U1U1Fermion.FdagF₊, U1U1Fermion.FdagF₋), (U1U1Fermion.FFdag₊, U1U1Fermion.FFdag₋), (U1U1Fermion.n₊, U1U1Fermion.n₋)),
    )
        modes = _fermion_modes(local_modes, L)
        direct, paired = InteractionTree(L), InteractionTree(L)
        reference = zeros(ComplexF64, size(modes[1][1]))
        for σ in eachindex(modes), i in 1:L
            T = σ == 1 ? hopping : conj(hopping) * 0.7
            name = Symbol("n", σ)
            addIntr!(direct, densities[σ], i, -T[i, i]; name)
            addIntr!(paired, densities[σ], i, -T[i, i]; name)
            reference .-= T[i, i] .* (modes[σ][i]' * modes[σ][i])
            for j in 1:L
                i == j && continue
                addIntr!(direct, FdagF[σ], (i, j), (true, true), -T[i, j]; Z = F.Z, name = (Symbol("Fd", σ), Symbol("F", σ)))
                reference .-= T[i, j] .* (modes[σ][i]' * modes[σ][j])
            end
            for j in i+1:L
                addIntr!(paired, FdagF[σ], (i, j), (true, true), -T[i, j]; Z = F.Z, name = (Symbol("Fd", σ), Symbol("F", σ)))
                addIntr!(paired, FFdag[σ], (i, j), (true, true), conj(T[i, j]); Z = F.Z, name = (Symbol("F", σ), Symbol("Fd", σ)))
            end
        end
        @test _dense_mpo(AutomataMPO(direct)) ≈ reference
        @test _dense_mpo(AutomataMPO(paired)) ≈ reference
    end
end

@testset "Spinless ordered products" begin
    F, L = U1SpinlessFermion, 4
    Ψ = randMPS(MersenneTwister(203), ComplexF64, F.pspace, [unitspace(F.pspace), F.pspace, fuse(F.pspace ⊗ F.pspace), F.pspace])
    ψ = _dense_mps(Ψ)
    local_mode = zeros(2, 2)
    local_mode[only(axes(F.pspace, U1Irrep(-1//2))), only(axes(F.pspace, U1Irrep(1//2)))] = 1
    c = only(_fermion_modes([local_mode], L))
    n = [f' * f for f in c]
    tree, cases = ObservableTree(L), []
    for sites in ((4, 1), (2, 2))
        i, j = sites
        push!(cases, (:hop, F.FdagF, sites, (true, true), c[i]' * c[j]))
        push!(cases, (:hole, F.FFdag, sites, (true, true), c[i] * c[j]'))
    end
    push!(cases, (:density, (F.n, F.n, F.n), (3, 1, 3), (false, false, false), n[3] * n[1] * n[3]))
    push!(cases, (:mixed, (F.FdagF..., F.n), (4, 1, 1), (true, true, false), c[4]' * c[1] * n[1]))
    for sites in ((4, 2, 1, 3), (1, 3, 3, 1), (1, 1, 3, 3))
        i, j, k, l = sites
        push!(cases, (:charge, (F.FdagF..., F.FdagF...), sites, (true, true, true, true), c[i]' * c[j] * c[k]' * c[l]))
        push!(cases, (:pair, F.ΔdagΔ, sites, (true, true, true, true), c[i]' * c[j]' * c[k] * c[l]))
    end
    for (name, ops, sites, parity, _) in cases
        addObs!(tree, ops, sites, parity; Z = F.Z, IntrName = name)
    end
    calObs!(tree, Ψ)
    values = convert(NamedTuple, tree)
    @testset "$name $sites" for (name, _, sites, _, reference) in cases
        @test getproperty(values, name)[sites] ≈ dot(ψ, reference * ψ) atol = 1e-12
    end
end

@testset "Non-Abelian ordered products" begin
    F, L = U1SU2Fermion, 4
    Ψ = randMPS(MersenneTwister(205), ComplexF64, F.pspace, [unitspace(F.pspace), F.pspace, fuse(F.pspace ⊗ F.pspace), F.pspace])
    ψ = _dense_mps(Ψ)
    up, down = zeros(4, 4), zeros(4, 4)
    empty = only(axes(F.pspace, Irrep[U₁×SU₂](-1, 0)))
    u, d = axes(F.pspace, Irrep[U₁×SU₂](0, 1//2))
    double = only(axes(F.pspace, Irrep[U₁×SU₂](1, 0)))
    up[empty, u] = up[d, double] = down[empty, d] = 1
    down[u, double] = -1
    a, b = _fermion_modes([up, down], L)
    n = [a[i]' * a[i] + b[i]' * b[i] for i in 1:L]
    sx = [(a[i]' * b[i] + b[i]' * a[i]) / 2 for i in 1:L]
    sy = [(a[i]' * b[i] - b[i]' * a[i]) / 2im for i in 1:L]
    sz = [(a[i]' * a[i] - b[i]' * b[i]) / 2 for i in 1:L]
    spin = (sx, sy, sz)
    hop(i, j) = a[i]' * a[j] + b[i]' * b[j]
    singlet(i, j) = (b[i] * a[j] - a[i] * b[j]) / sqrt(2)
    triplet(i, j) = (a[i] * a[j], (a[i] * b[j] + b[i] * a[j]) / sqrt(2), b[i] * b[j])
    spinbond(i, j) = ((a[i]' * b[j] + b[i]' * a[j]) / 2, (a[i]' * b[j] - b[i]' * a[j]) / 2im, (a[i]' * a[j] - b[i]' * b[j]) / 2)
    tree, cases = ObservableTree(L), []
    for sites in ((4, 1), (2, 2))
        i, j = sites
        push!(cases, (:hop, F.FdagF, sites, (true, true), hop(i, j)))
        push!(cases, (:hole, F.FFdag, sites, (true, true), a[i] * a[j]' + b[i] * b[j]'))
        push!(cases, (:spin, F.SS, sites, (false, false), sum(s[i] * s[j] for s in spin)))
    end
    for sites in ((3, 1, 2), (2, 2, 4))
        i, j, k = sites
        reference = sx[i] * sy[j] * sz[k] + sy[i] * sz[j] * sx[k] + sz[i] * sx[j] * sy[k] - sz[i] * sy[j] * sx[k] - sy[i] * sx[j] * sz[k] - sx[i] * sz[j] * sy[k]
        push!(cases, (:chiral, F.SSS, sites, (false, false, false), reference / im))
    end
    push!(cases, (:mixed, (F.FdagF..., F.n), (4, 1, 1), (true, true, false), hop(4, 1) * n[1]))
    push!(cases, (:density, (F.n, F.n, F.n, F.n), (1, 3, 1, 3), (false, false, false, false), n[1] * n[3] * n[1] * n[3]))
    for sites in ((4, 2, 1, 3), (1, 3, 3, 1), (1, 1, 3, 3))
        i, j, k, l = sites
        push!(cases, (:singlet, F.ΔₛdagΔₛ, sites, (true, true, true, true), singlet(j, i)' * singlet(k, l)))
        push!(cases, (:triplet, F.ΔₜdagΔₜ, sites, (true, true, true, true), sum(x' * y for (x, y) in zip(triplet(j, i), triplet(l, k)))))
        push!(cases, (:charge, (F.FdagF..., F.FdagF...), sites, (true, true, true, true), hop(i, j) * hop(k, l)))
        push!(cases, (:spinbond, F.SBSB, sites, (true, true, true, true), sum(x * y for (x, y) in zip(spinbond(i, j), spinbond(k, l)))))
    end
    for (name, ops, sites, parity, _) in cases
        addObs!(tree, ops, sites, parity; Z = F.Z, IntrName = name)
    end
    calObs!(tree, Ψ)
    values = convert(NamedTuple, tree)
    @testset "$name $sites" for (name, _, sites, _, reference) in cases
        @test getproperty(values, name)[sites] ≈ dot(ψ, reference * ψ) atol = 1e-12
    end
end
