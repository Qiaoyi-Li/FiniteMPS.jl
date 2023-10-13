"""
     struct LanczosInfo 
          converged::Bool
          normres::Vector{Float64}    
          numiter::Int64    
          numops::Int64
     end

Similar to `KrylovKit.ConvergenceInfo` but delete `residuals` to save memory.
"""
struct LanczosInfo
     converged::Bool
     normres::Vector{Float64}
     numiter::Int64
     numops::Int64
end
function convert(::Type{LanczosInfo}, Info::KrylovKit.ConvergenceInfo)
     return LanczosInfo(Info.converged > 0, Info.normres, Info.numiter, Info.numops)
end

"""
     struct DMRG2Info
          Eg::Float64
          Lanczos::LanczosInfo 
          TrunErr::Float64
     end

Information of 2-site DMRG.
"""
struct DMRG2Info
     Eg::Float64
     Lanczos::LanczosInfo
     TrunErr::Float64
end

"""
     struct DMRG1Info
          Eg::Float64
          Lanczos::LanczosInfo 
     end

Information of 1-site DMRG.
"""
struct DMRG1Info
     Eg::Float64
     Lanczos::LanczosInfo
end

"""
     DMRGSweep2!(Env::SparseEnvironment{L,3,T}; kwargs...)

2-site DMRG sweep from left to right and sweep back from right to left.  

# Kwargs
     trunc::TruncationScheme = notrunc()
Control the truncation in svd after each 2-site update. Details see `tsvd`. 

     GCstep::Bool = false
`GC.gc()` manually after each step if `true`.

     GCsweep::Bool = false 
`GC.gc()` manually after each (left to right or right to left) sweep if `true`.   
"""
function DMRGSweep2!(Env::SparseEnvironment{L,3,T}; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}

     trunc = get(kwargs, :trunc, notrunc())
     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)

     Ψ = Env[3]
     canonicalize!(Ψ, 1, 2)

     # info, (L, R)
     info = (Vector{DMRG2Info}(undef, L - 1), Vector{DMRG2Info}(undef, L - 1))
     # left to right sweep
     Al::MPSTensor = Ψ[1]
     for si = 1:L-1
          @timeit GlobalTimer "pushEnv" canonicalize!(Env, si, si + 1)
          Ar = Ψ[si+1]

          @timeit GlobalTimer "DMRGUpdate2" eg, xg, info_Lanczos = _DMRGUpdate2(ProjHam(Env, si, si + 1), Al, Ar; kwargs...)
          @timeit GlobalTimer "svd" Ψ[si], s, vd, ϵ = tsvd(xg; trunc=trunc)
          # next Al
          Al = s * vd
          # remember to change Center of Ψ manually
          Center(Ψ)[:] = [si + 1, si + 1]
          info[1][si] = DMRG2Info(eg, info_Lanczos, ϵ)

          # GC manually
          GCstep && @everywhere GC.gc()
     end
     Ψ[L] = Al
     canonicalize!(Ψ, L - 1)

     # GC manually
     GCsweep && @everywhere GC.gc()

     # right to left sweep, skip [L-1, L]
     Ar::MPSTensor = Ψ[L-1]
     for si = reverse(2:L-1)
          @timeit GlobalTimer "pushEnv" canonicalize!(Env, si - 1, si)
          Al = Ψ[si-1]

          @timeit GlobalTimer "DMRGUpdate2" eg, xg, info_Lanczos = _DMRGUpdate2(ProjHam(Env, si - 1, si), Al, Ar; kwargs...)
          @timeit GlobalTimer "svd" u, s, Ψ[si], ϵ = tsvd(xg; trunc=trunc)
          # next Ar
          Ar = u * s
          # remember to change Center of Ψ manually
          Center(Ψ)[:] = [si - 1, si - 1]

          info[2][si-1] = DMRG2Info(eg, info_Lanczos, ϵ)

          # GC manually
          GCstep && @everywhere GC.gc()
     end
     Ψ[1] = Ar

     # GC manually
     GCsweep && @everywhere GC.gc()

     return info

end

"""
     DMRGSweep1!(Env::SparseEnvironment{L,3,T}; kwargs...)

1-site DMRG sweep from left to right and sweep back from right to left.  

# Kwargs
     GCstep::Bool = false
`GC.gc()` manually after each step if `true`.

     GCsweep::Bool = false 
`GC.gc()` manually after each (left to right or right to left) sweep if `true`.   
"""
function DMRGSweep1!(Env::SparseEnvironment{L,3,T}; kwargs...) where {L,T<:Tuple{AdjointMPS,SparseMPO,MPS}}

     GCstep = get(kwargs, :GCstep, false)
     GCsweep = get(kwargs, :GCsweep, false)

     Ψ = Env[3]

     # info, (L, R)
     info = (Vector{DMRG1Info}(undef, L), Vector{DMRG1Info}(undef, L))
     # left to right sweep
     for si = 1:L
          canonicalize!(Ψ, si)
          canonicalize!(Env, si)

          eg, Ψ[si], info_Lanczos = _DMRGUpdate1(ProjHam(Env, si), Ψ[si]; kwargs...)
          info[1][si] = DMRG1Info(eg, info_Lanczos)

          # GC manually
          GCstep && @everywhere GC.gc()
     end

     # GC manually
     GCsweep && @everywhere GC.gc()

     # right to left sweep, skip L
     for si = reverse(1:L-1)
          canonicalize!(Ψ, si)
          canonicalize!(Env, si)

          eg, Ψ[si], info_Lanczos = _DMRGUpdate1(ProjHam(Env, si), Ψ[si]; kwargs...)
          info[2][si] = DMRG1Info(eg, info_Lanczos)

          # GC manually
          GCstep && @everywhere GC.gc()
     end

     # GC manually
     GCsweep && @everywhere GC.gc()

     return info

end

function _DMRGUpdate2(H::SparseProjectiveHamiltonian{2}, Al::MPSTensor, Ar::MPSTensor; kwargs...)
     # 2-site update of DMRG 

     x2 = CompositeMPSTensor(Al, Ar)
     @timeit GlobalTimer "LanczosGS2" eg, xg, info = eigsolve(x -> action2(H, x; kwargs...), x2, 1, :SR, _getLanczos(; kwargs...))

     return eg[1], xg[1], info

end

function _DMRGUpdate1(H::SparseProjectiveHamiltonian{1}, A::MPSTensor; kwargs...)
     # 1-site update of DMRG

     @timeit GlobalTimer "LanczosGS1" eg, xg, info = eigsolve(x -> action1(H, x; kwargs...), A, 1, :SR, _getLanczos(; kwargs...))
     return eg[1], xg[1], info

end