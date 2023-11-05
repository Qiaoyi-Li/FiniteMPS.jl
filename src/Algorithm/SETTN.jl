"""
     SETTN(H::SparseMPO, β::Number; kwargs...) -> ρ::MPO, lsF::Vector{Float64}

Use series-expansion thermal tensor network (SETTN)`[https://doi.org/10.1103/PhysRevB.95.161104]` method to initialize a high-temperature MPO `ρ = e^(-βH/2)`. Note `ρ` is unnormalized. The list of free energy `F = -lnTr[ρρ^†]/β` with different expansion orders is also returned.

# Kwargs
     disk::Bool = false
     maxorder::Int64 = 4
     bspace::VectorSpace (details please see identityMPO)

Note we use `mul!` and `axpby!` to implement `H^n -> H^(n+1)` and `ρ -> ρ + (-βH/2)^n / n!`, respectively. All kwargs of these two functions are valid and will be propagated to them appropriately.
"""
function SETTN(H::SparseMPO{L}, β::Number; kwargs...) where {L}
     @assert β ≥ 0

     maxorder::Int64 = get(kwargs, :maxorder, 4)
     verbose::Int64 = get(kwargs, :verbose, 0)
     tol::Float64 = get(kwargs, :tol, 1e-8)


     # deduce pspace 
     lspspace = map(H) do M
          idx = findfirst(x -> !isnothing(x) && !isa(x, IdentityOperator), M)
          domain(M[idx])[1]
     end

     Hn = identityMPO(L, lspspace; kwargs...) # H0 = Id
     ρ = identityMPO(L, lspspace; kwargs...)

     lsF = zeros(Float64, maxorder)
     for n = 1:maxorder
          # Hn -> H * Hn
          mul!(Hn, H, deepcopy(Hn), 1, 0; kwargs...)
          # ρ -> ρ + Hn * (-β/2)^n/ n!     
          axpy!((-β / 2)^n / factorial(n), Hn, ρ; kwargs...)
          lsF[n] = -norm(ρ)^2 / (β)

          manualGC()

          n > 1 && (δF = (lsF[n] - lsF[n-1]) / abs(lsF[n]))
          if verbose ≥ 1
               dF_str = n > 1 ? ", δF/|F| = $(δF)" : ""
               println("SETTN order = $(n), F = $(lsF[n])", dF_str)
          end

          if n > 1 && abs(δF) < tol
               println("SETTN converged at order = $(n), δF/|F| = $(δF) (tol = $(tol))!")
               return ρ, lsF[1:n]
          end
     end

     return ρ, lsF

end