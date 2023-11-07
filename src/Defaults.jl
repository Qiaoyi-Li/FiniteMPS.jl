module KrylovDefault
     using KrylovKit

     krylovdim = 16
     maxiter = 1
     tol = 1e-8
     orth=ModifiedGramSchmidt()
end

function _getLanczos(; kwargs...)
     LanczosOpt = get(kwargs, :LanczosOpt,  NamedTuple())
     
     return Lanczos(;
          krylovdim=get(LanczosOpt, :krylovdim, KrylovDefault.krylovdim),
          maxiter=get(LanczosOpt, :maxiter, KrylovDefault.maxiter),
          tol=get(LanczosOpt, :tol, KrylovDefault.tol),
          orth=get(LanczosOpt, :orth, KrylovDefault.orth),
          eager=true, verbosity=0)
end

module MPSDefault
     tol = 1e-8 
end


