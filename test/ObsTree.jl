# test if ObsTree work as expected

@testset "NoSym" verbose = true begin
	L = 6
	lsθ = rand(L) * 2π
	lsSz = map(x -> (cos(x)^2 - sin(x)^2) / 2, lsθ)
	# generate a product state
	Ψ = MPS(map(lsθ) do θ
		TensorMap([cos(θ), sin(θ)], ℂ^1 ⊗ ℂ^2, ℂ^1)
	end
	)
	canonicalize!(Ψ, 1)
	OpSz = TensorMap([0.5 0.0; 0.0 -0.5], ℂ^2, ℂ^2)

	Tree = ObservableTree()
	for i in 1:L, j in 1:L, k in 1:L, l in 1:L
		!allunique([i, j, k, l]) && continue
		addObs!(Tree, Tuple(fill(OpSz, 4)), (i, j, k, l), (false, false, false, false); name = (:Sz, :Sz, :Sz, :Sz))
	end

	for i in 1:L, j in 1:L, k in 1:L
		!allunique([i, j, k]) && continue
		addObs!(Tree, Tuple(fill(OpSz, 3)), (i, j, k), (false, false, false); name = (:Sz, :Sz, :Sz))
	end

	for i in 1:L, j in 1:L
		i == j && continue
		addObs!(Tree, Tuple(fill(OpSz, 2)), (i, j), (false, false); name = (:Sz, :Sz))
	end

	for i in 1:L
		addObs!(Tree, OpSz, i; name = :Sz)
	end

	calObs!(Tree, Ψ)
	Obs = convert(NamedTuple, Tree)

	@testset "1-site" begin
		for (k, v) in Obs.Sz
			@test v ≈ lsSz[k[1]]
		end
	end

	@testset "2-site" begin
		for (k, v) in Obs.SzSz
			@test v ≈ lsSz[k[1]] * lsSz[k[2]]
		end
	end

	@testset "3-site" begin
		for (k, v) in Obs.SzSzSz
			@test v ≈ lsSz[k[1]] * lsSz[k[2]] * lsSz[k[3]]
		end
	end

	@testset "4-site" begin
		for (k, v) in Obs.SzSzSzSz
			@test v ≈ lsSz[k[1]] * lsSz[k[2]] * lsSz[k[3]] * lsSz[k[4]]
		end
	end

end


