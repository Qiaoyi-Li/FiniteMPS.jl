"""
	LanczosExp(f::Function, x₀, t::Number, args...;
		K::Int64 = 32,
		tol::Real = 1e-8,
		callback::Union{Nothing, Function} = nothing,
		verbose = false
	) -> exp(At)*x, info::Dict{Symbol, Any}

Compute `exp(At)x₀` for a hermitian map `A: x -> f(x, args...)` using the Lanczos algorithm. `f` can be any function as if it acts on `x` like an hermitian operator and preserves the type of `x`.
	
# Required methods	
`x` can be any type as if it behaves like a vector in Hilbert space, i.e. the following methods are implemented:

	eltype(x)
Return the element type of `x`, e.g. `Float64` or `ComplexF64`.

	normalize!(x)
In-place normalize `x` according to the inner-induced norm.

	norm(x)
Return the inner-induced norm.

	inner(x, y)
Return the inner product `⟨x, y⟩`.

	add!(y, x, α)
In-place add `y -> y + αx`.

	rmul!(x, α)
In-place multiply `x -> αx`.

# Kwargs 
	K::Int64 = 32
Krylov space dimension.

	tol::Real = 1e-8
Convergence tolerance.  

	callback::Union{Nothing, Function} = nothing
A in-placed callback function, applied to x after each iteration.

	verbose::Bool = false
If `true`, print convergence information.
"""
function LanczosExp(f::Function, x₀, t::NT, args...;
	K::Int64 = 32,
	tol::Real = 1e-8,
	callback::Union{Nothing, Function} = nothing,
	verbose::Bool = false) where NT

	T = zeros(K + 1, K + 1)  # tridiagonal matrix
	lsb = Vector{Any}(undef, K + 1) # Lanczos vectors

	# first one
	norm0 = norm(x₀)
	lsb[1] = rmul!(deepcopy(x₀), 1 / norm0)
	Vt = zeros(promote_type(NT, Float64), K)
	for k in 1:K
		# A * bₖ
		lsb[k+1] = f(lsb[k], args...)

		# ⟨bₖ| A |bₖ⟩
		T[k, k] = real(inner(lsb[k], lsb[k+1]))

		# orthgonalize
		# bₖ₊₁ = bₖ₊₁ - ⟨bₖ|bₖ₊₁⟩bₖ - ⟨bₖ₋₁|bₖ⟩bₖ₋₁
		add!(lsb[k+1], lsb[k], -T[k, k])
		k > 1 && add!(lsb[k+1], lsb[k-1], -T[k-1, k])

		T[k, k+1] = T[k+1, k] = norm(lsb[k+1])

		# normalize
		rmul!(lsb[k+1], 1 / T[k, k+1])

		# callback function here
		!isnothing(callback) && callback(lsb[k+1])

		# convergence check 
		ϵ, V = eigen(T[1:k, 1:k])
		expT = V * diagm(exp.(ϵ * t)) * V'

		if k ≤ 2
			err2 = Inf
		else
			err2 = max(norm(expT[:, 1] - Vt[1:k]), norm(expT[end-1:end, 1]))^2 / norm(expT[:, 1])^2
		end
		copyto!(Vt, expT[:, 1])
		if err2 < tol^2 # converged eigen vector
			verbose && println("eigen vector converged, err2 = $(err2), break at K = $(k)!")
			break
		end
		# T[k, k+1] = ⟨bₖ|A|bₖ₊₁⟩, scale by the estimated eigval so that A -> a*A give a similar cutoff
		ϵmax = maximum(abs, ϵ)
		if T[k, k+1] ≤ tol * ϵmax # "=" cannot be skipped!  
			# closed subspace
			verbose && println("T[$k, $(k+1)]/max|ϵ| = $(T[k, k+1]/ϵmax), break at K = $(k)!")
			break
		end

	end
	rmul!(Vt, norm0)

	# linear combination
	if eltype(x₀) != eltype(Vt)
		expAx = Vt[1] * lsb[1]
	else
		# warning: if Lanczos basis is needed, use * here
		expAx = rmul!(lsb[1], Vt[1])
	end
	K_cut = findlast(i -> isassigned(lsb, i), 1:K+1) - 1
	for k in 2:K_cut
		add!(expAx, lsb[k], Vt[k])
	end

	info = (V = Vt[1:K_cut], T = T[1:K_cut, 1:K_cut])

	return expAx, info
end
