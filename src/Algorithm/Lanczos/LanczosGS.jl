"""
	LanczosGS(f::Function, x₀, args...;
		K::Int64 = 32,
		tol::Real = 1e-8,
		callback::Union{Nothing, Function} = nothing,
		verbose = false
	) -> ϵg::Number, xg::AbstractVector, info::Dict{Symbol, Any}

Solve the ground state problem for a hermitian map `x -> f(x, args...)` using the Lanczos algorithm. `f` can be any function as if it acts on `x` like an hermitian operator and preserves the type of `x`.
	
# Required methods	
`x` can be any type as if it behaves like a vector in Hilbert space, i.e. the following methods are implemented:

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
function LanczosGS(f::Function, x₀, args...;
	K::Int64 = 32,
	tol::Real = 1e-8,
	callback::Union{Nothing, Function} = nothing,
     verbose::Bool = false)

	T = zeros(K + 1, K + 1)  # tridiagonal matrix
	lsb = Vector{Any}(undef, K + 1) # Lanczos vectors

	# first one
	lsb[1] = normalize!(deepcopy(x₀))
	Vg = zeros(K)
     ϵg = fill(NaN, K)
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
          !isnothing(callback) && callback(x)

		# convergence check 
		ϵ, V = eigen(T[1:k, 1:k])
		if k ≤ 2
			err2 = Inf 
		else
			err2 = max(norm(V[:, 1] - Vg[1:k])^2, norm(V[end-1:end, 1])^2)
		end
		copyto!(Vg, V[:, 1])
          ϵg[k] = ϵ[1]
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
	
     # linear combination
	xg = rmul!(lsb[1], Vg[1])
     K_cut = findlast(!isnan, ϵg)
	for k in 2:K_cut
		add!(xg, lsb[k], Vg[k])
	end

	info = (V = Vg[1:K_cut], ϵ = ϵg[1:K_cut], T = T[1:K_cut, 1:K_cut])

     return ϵg[K_cut], xg, info
end