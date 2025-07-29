"""
	mul!(C::DenseMPS, A::SparseMPO, B::DenseMPS, α::Number, β::Number; kwargs...) 

Compute `C = α A*B + β C` variationally via 2-site update, where `A` is a sparse MPO, `B` and `C` are dense MPS/MPO. Note 'B' cannot reference to the same MPS/MPO with `C`.  

	mul!(C::DenseMPS, A::SparseMPO, B::DenseMPS; kwargs...)

Compute `C = A*B` by letting `α = 1` and `β = 0`.

# Kwargs
	 trunc::TruncationScheme = truncbelow(MPSDefault.tol) & truncdim(MPSDefault.D)
	 GCstep::Bool = false
	 GCsweep::Bool = false
	 maxiter::Int64 = 8
	 disk::Bool = false
	 tol::Float64 = 1e-8
	 verbose::Int64 = 0
	 CBEAlg::CBEAlgorithm = NoCBE()
	 lsnoise::AbstractVector{Float64} = Float64[]

Note CBE can only be used when `α ≠ 0`.
"""
function mul!(C::DenseMPS{L}, A::SparseMPO, B::DenseMPS{L}, α::Number, β::Number; kwargs...) where {L}
	@assert α != 0 || β != 0
	@assert !(B === C)

	trunc::TruncationScheme = get(kwargs, :trunc, truncbelow(MPSDefault.tol) & truncdim(MPSDefault.D))
	GCstep::Bool = get(kwargs, :GCstep, false)
	GCsweep::Bool = get(kwargs, :GCsweep, false)
	maxiter::Int64 = get(kwargs, :maxiter, 8)
	tol::Float64 = get(kwargs, :tol, 1e-8)
	verbose::Int64 = get(kwargs, :verbose, 0)
	lsnoise::Vector{Float64} = get(kwargs, :lsnoise, Float64[])
	CBEAlg = get(kwargs, :CBEAlg, NoCBE())
	!isa(CBEAlg, NoCBE) && @assert !iszero(α)

	if α != 0
		Env_mul = Environment(C', A, B; kwargs...)
		canonicalize!(Env_mul, 1)
	end
	if β != 0
		C₀ = deepcopy(C)
		Env_add = Environment(C', C₀; kwargs...)
		canonicalize!(Env_add, 1)
	end

	canonicalize!(C, 1)
	@assert coef(C) != 0
	if isa(CBEAlg, NoCBE)
		# 2-site sweeps
		for iter ∈ 1:maxiter
			TimerSweep = TimerOutput()
			direction = SweepL2R
			convergence::Float64 = 0
			lsinfo = BondInfo[]
			@timeit TimerSweep "Sweep2" for si ∈ vcat(1:L-1, reverse(1:L-1))

				TimerStep = TimerOutput()
				# 2-site local tensor before update
				x₀ = rmul!(CompositeMPSTensor(C[si], C[si+1]), coef(C))

				if α != 0
					@timeit TimerStep "pushEnv_mul" canonicalize!(Env_mul, si, si + 1)
					@timeit TimerStep "action2_mul" ab = action2(ProjHam(Env_mul, si, si + 1), B[si], B[si+1]; kwargs...)
					_α = α * coef(B) / coef(C)
				else
					ab = nothing
					_α = 0
				end
				if β != 0
					@timeit TimerStep "pushEnv_add" canonicalize!(Env_add, si, si + 1)
					@timeit TimerStep "action2_add" c = action2(ProjHam(Env_add, si, si + 1), C₀[si], C₀[si+1]; kwargs...)
					_β = β * coef(C₀) / coef(C)
				else
					c = nothing
					_β = 0
				end

				x = axpby!(_α, ab, _β, c)
				# normalize
				norm_x = norm(x)
				rmul!(x, 1 / norm_x)
				C.c *= norm_x

				# apply noise
				iter ≤ length(lsnoise) && noise!(x, lsnoise[iter])

				# svd
				if direction <: SweepL2R
					@timeit TimerStep "svd" C[si], C[si+1], svdinfo = leftorth(x; trunc = trunc, kwargs...)
					Center(C)[:] = [si + 1, si + 1]
				else
					@timeit TimerStep "svd" C[si], C[si+1], svdinfo = rightorth(x; trunc = trunc, kwargs...)
					Center(C)[:] = [si, si]
				end
				push!(lsinfo, svdinfo)

				# check convergence
				@timeit TimerStep "convergence_check" begin
					x = rmul!(CompositeMPSTensor(C[si], C[si+1]), coef(C))
					convergence = max(convergence, norm(x - x₀)^2 / abs2(coef(C)))
				end
				# GC manually
				GCstep && manualGC(TimerStep)

				merge!(TimerSweep, TimerStep; tree_point = ["Sweep2"])
				if verbose ≥ 2
					ar = direction <: SweepL2R ? "->" : "<-"
					show(TimerStep; title = "site $(si) $(ar) $(si+1)")
					println("\niter $(iter), site $(si) $(ar) $(si+1), $(lsinfo[end]), max convergence = $(convergence)")
					flush(stdout)
				end

				# change direction
				si == L - 1 && (direction = SweepR2L)

			end

			GCsweep && manualGC(TimerSweep)
			if verbose ≥ 1
				show(TimerSweep; title = "mul! iter $(iter)")
				println("\niter $(iter), $(merge(lsinfo)), max convergence = $(convergence) (tol = $(tol))")
				flush(stdout)
			end

			if iter > length(lsnoise) && convergence < tol
				break
			end

		end
	else
		# 1-site sweeps with CBE
		for iter ∈ 1:maxiter
			TimerSweep = TimerOutput()
			direction = SweepL2R
			convergence::Float64 = 0
			lsinfo = BondInfo[]
			@timeit TimerSweep "Sweep1" for si ∈ vcat(1:L-1, reverse(1:L))

				TimerStep = TimerOutput()

				# CBE
				if direction <: SweepL2R
					si_L, si_R = si, si + 1
				else
					si_L, si_R = si - 1, si
				end


				if si_L > 0
					canonicalize!(Env_mul, si_L, si_R)
					@timeit TimerStep "CBE" C[si_L], C[si_R], info_cbe, TO_CBE = CBE(C[si_L], C[si_R], Env_mul.El[si_L], Env_mul.Er[si_R], Env_mul[2][si_L], Env_mul[2][si_R], convert(CBEAlgorithm{direction}, CBEAlg);
						Bl = B[si_L], Br = B[si_R],
					)

					merge!(TimerStep, TO_CBE; tree_point = ["CBE"])

				end

				# local tensor before update
				x₀ = rmul!(C[si], coef(C))


				@timeit TimerStep "pushEnv_mul" canonicalize!(Env_mul, si)
				PH = CompositeProjectiveHamiltonian(Env_mul.El[si], Env_mul.Er[si], (Env_mul[2][si],))
				@timeit TimerStep "action_mul" ab = action(B[si], PH, TimerStep)
				finalize(PH)

				_α = α * coef(B) / coef(C)

				if β != 0
					si_L > 0 && canonicalize!(Env_add, si_L, si_R) # Er[si] is incorrect after CBE
					@timeit TimerStep "pushEnv_add" canonicalize!(Env_add, si)
					pspace = codomain(x₀)[2]
					PH = SimpleProjectiveHamiltonian(Env_add.El[si], Env_add.Er[si], IdentityOperator(pspace, trivial(pspace), si, 1.0))
					@timeit TimerStep "action_add" c = action(C₀[si], PH, TimerStep)
					finalize(PH)
					_β = β * coef(C₀) / coef(C)
				else
					c = nothing
					_β = 0
				end

				x = axpby!(_α, ab, _β, c)
				# normalize
				norm_x = norm(x)
				rmul!(x, 1 / norm_x)
				C.c *= norm_x

				# apply noise
				if iter ≤ length(lsnoise)
					axpby!(lsnoise[iter], randn(eltype(x.A), codomain(x.A), domain(x.A)), 1.0, x.A)
					normalize!(x.A)
				end

				# check convergence before svd
				@timeit TimerStep "convergence_check" begin
					convergence = max(convergence, norm(x * coef(C) - x₀)^2 / abs2(coef(C)))
				end

				# svd
				if direction <: SweepL2R
					@timeit TimerStep "svd" C[si], S, svdinfo = leftorth(x; trunc = trunc, kwargs...)
					C[si+1] = MPSTensor(S) * C[si+1]
					Center(C)[:] = [si + 1, si + 1]
				elseif si > 1
					@timeit TimerStep "svd" S, C[si], svdinfo = rightorth(x; trunc = trunc, kwargs...)
					C[si-1] = C[si-1] * MPSTensor(S)
					Center(C)[:] = [si - 1, si - 1]
				else
					C[si] = x
				end
				si_L > 0 && push!(lsinfo, svdinfo)

				# GC manually
				GCstep && manualGC(TimerStep)

				merge!(TimerSweep, TimerStep; tree_point = ["Sweep1"])
				if verbose ≥ 2
					ar = direction <: SweepL2R ? "->" : "<-"
					show(TimerStep; title = "site $(si) $(ar)")
					println("\niter $(iter), site $(si) $(ar), $(lsinfo[end]), max convergence = $(convergence)")
					flush(stdout)
				end

				# change direction
				si == L - 1 && (direction = SweepR2L)

			end

			GCsweep && manualGC(TimerSweep)
			if verbose ≥ 1
				show(TimerSweep; title = "mul! iter $(iter)")
				println("\niter $(iter), $(merge(lsinfo)), max convergence = $(convergence) (tol = $(tol))")
				flush(stdout)
			end

			if iter > length(lsnoise) && convergence < tol
				break
			end

		end

	end

	return C

end

function mul!(C::DenseMPS, A::SparseMPO, B::DenseMPS; kwargs...)
	# C = A*B
	return mul!(C, A, B, 1, 0; kwargs...)
end

"""
	rmul!(A::DenseMPS, b::Number)

In-place multiplication `A -> b*A` where `b` is a scalar.
"""
function rmul!(A::DenseMPS, b::Number)
	# A -> b*A
	A.c *= b
	return A
end
