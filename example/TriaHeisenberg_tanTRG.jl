using MKL
using FiniteMPS, FiniteLattices

include("Models/Heisenberg.jl")

verbose = 1
GCstep = false
@show Latt = YCTriangular(12, 6, SnakePath)
D = 512
Para = (J=1, J′=0)

lsβ = vcat(2.0 .^ (-15:2:-1), 1:16)
lsF = zeros(length(lsβ))
lsE = zeros(length(lsβ))

let

     H = AutomataMPO(SU₂Heisenberg(Latt; Para...))

     # ============= SETTN ===============
     ρ, lsF_SETTN = SETTN(H, lsβ[1];
          maxorder=4, verbose=1,
          GCstep=false, GCsweep=true, tol=1e-12,
          compress = 1e-15,
          trunc=truncdim(128))
     lsF[1] = lsF_SETTN[end]
     lnZ = 2 * log(norm(ρ))
     normalize!(ρ)
     # ==================================

     Env = Environment(ρ', H, ρ)
     canonicalize!(Env, 1)
     lsE[1] = scalar!(Env; normalize=true)

     for i = 2:length(lsβ)
          dβ = lsβ[i] - lsβ[i-1]

          TDVPSweep2!(Env, -dβ / 2;
               GCstep=GCstep, GCsweep=true, verbose=verbose,
               trunc=truncdim(D) & truncbelow(1e-16))

          # TDVPSweep1!(Env, -dβ / 2;
          #      CBEAlg = StandardCBE(D + div(D, 8), 1e-8),
          #      GCstep=GCstep, GCsweep=true, verbose=verbose,
          #      trunc=truncdim(D) & truncbelow(1e-16))


          lnZ += 2 * log(norm(ρ))
          normalize!(ρ)

          lsF[i] = -lnZ / lsβ[i]
          lsE[i] = scalar!(Env; normalize=true)

          let 
               println("β = $(lsβ[i]), F = $(lsF[i]), E = $(lsE[i])")
               flush(stdout)
          end
     end

end



