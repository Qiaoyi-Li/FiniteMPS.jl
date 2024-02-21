global DMRGDefaultLanczos = Lanczos(;
     krylovdim = 8,
     maxiter = 1,
     tol = 1e-8,
     orth = ModifiedGramSchmidt(),
     eager = true,
     verbosity = 0
)

global TDVPDefaultLanczos = Lanczos(;
     krylovdim = 32,
     maxiter = 1,
     tol = 1e-8,
     orth = ModifiedGramSchmidt(),
     eager = true,
     verbosity = 0
)

"""
module MPSDefault
     D::Int64 = 512
     tol::Float64 = 1e-8
end

Default maximum bond dimension `D` and truncation tolerance `tol` for MPS, used as default truncation scheme in some 2-site update algorithms.  
"""
module MPSDefault
     D::Int64 = 512
     tol::Float64 = 1e-8
end



