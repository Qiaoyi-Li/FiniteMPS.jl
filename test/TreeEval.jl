@testset "shared tree evaluation" begin
    tree = ObservableTree(4)
    refs = Dict{Tuple{Int}, Ref{Number}}()
    tree.Refs["products"] = refs
    sequences = ([2, 3, 5, 7], [2, 3, 11, 7], [2, 3, 5, 7])
    for (i, sequence) in enumerate(sequences)
        refs[(i,)] = Ref{Number}()
        FiniteMPS._insert_paired_channel!(tree, sequence, refs[(i,)]; phase = i == 3 ? -1 : 1)
    end
    merge!(tree)
    plan = FiniteMPS._tree_plan(tree)
    @test sum(length, plan.joins) == 2
    pushes = Threads.Atomic{Int}(0)
    pushenv(_, node, env, _) = (Threads.atomic_add!(pushes, 1); env * node.Op[2])
    for (disk, capacity, workers) in ((false, 0, 1), (true, 0, 1), (true, 1, Threads.nthreads(:default)))
        pushes[] = 0
        FiniteMPS._evaluate_tree!(tree, (side, site) -> nothing, pushenv, 1, 1,
            LayeredTreeEval(ntasks = workers); disk, maxsize = capacity, verbose = 0,
            showtimes = 10, GCspacing = 0)
        @test [refs[(i,)][] for i in 1:3] == [210, 462, -210]
        @test pushes[] == length(plan.nodes) - 2
    end
    @test_throws ErrorException FiniteMPS._evaluate_tree!(tree, (side, site) -> nothing,
        (args...) -> error("push failed"), 1, 1, LayeredTreeEval();
        disk = true, maxsize = 0, verbose = 0, showtimes = 10, GCspacing = 0)

    store = FiniteMPS._TreeEnvStore([2, 1], true, 1)
    try
        value = [1, 2, 3]
        FiniteMPS._tree_publish!(store, 1, value)
        held = FiniteMPS._tree_borrow!(store, 1)
        FiniteMPS._tree_publish!(store, 2, [4])
        @test !store.entries[1].cached
        shared = FiniteMPS._tree_borrow!(store, 1)
        @test shared === held === value
        FiniteMPS._tree_return!(store, 1)
        FiniteMPS._tree_return!(store, 1; consumed = true)
        filename = FiniteMPS._tree_filename(store, 1)
        @test isfile(filename)
        loaded = FiniteMPS._tree_borrow!(store, 1)
        @test loaded == value
        @test isfile(filename)
        FiniteMPS._tree_return!(store, 1; consumed = true)
        @test !isfile(filename)
    finally
        rm(store.directory; recursive = true, force = true)
    end
end

@testset "paired ITP registration" begin
    p, v = ℂ^2, ℂ^1
    density = TensorMap([1.0 0.0; 0.0 2.0], p, p)
    site = permute(id(v) ⊗ density, ((1, 2), (4, 3)))
    rho = MPO([site, site])
    sz = TensorMap([0.5 0.0; 0.0 -0.5], p, p)
    tree = ImagTimeProxyTree(2)
    addITP!(tree, ((sz, sz), (sz, sz)), (2, 1, 1, 2), (false, false, false, false);
        name = (:Sz, :Sz, :Sz, :Sz))
    addITP!(tree, ((sz, sz), (sz, sz)), (1, 2, 1, 2), (false, false, false, false);
        name = (:Sz, :Sz, :Sz, :Sz))
    firstref = tree.Refs["SzSz_SzSz"][(2, 1, 1, 2)]
    addITP!(tree, ((sz, sz), (sz, sz)), (2, 1, 1, 2), (false, false, false, false);
        name = (:Sz, :Sz, :Sz, :Sz))
    @test tree.Refs["SzSz_SzSz"][(2, 1, 1, 2)] === firstref
    for (disk, alg) in ((false, LayeredTreeEval(ntasks = 1)), (true, LayeredTreeEval()))
        calITP!(tree, rho; disk, maxsize = 0, alg)
        @test all(ref -> ref[] ≈ 1.25^2, values(tree.Refs["SzSz_SzSz"]))
    end
    @test sum(length, FiniteMPS._tree_plan(tree).joins) == 1
    @test_throws ArgumentError addITP!(tree, ((), (sz,)), (1,), (false,))
end

@testset "ITP auxiliary spin channel" begin
    p = SU2Spin.pspace
    v = unitspace(p)
    site = permute(id(v) ⊗ id(p), ((1, 2), (4, 3)))
    rho = MPO([site, site])
    SL, SR = SU2Spin.SS
    tree = ImagTimeProxyTree(2)
    for sites in ((1, 1), (1, 2))
        addITP!(tree, ((SL,), (SR,)), sites, (false, false); name = (:S, :S))
    end
    calITP!(tree, rho; disk = true, maxsize = 0)
    @test [tree.Refs["S_S"][sites][] for sites in ((1, 1), (1, 2))] ≈ [3.0, 0.0]
end
