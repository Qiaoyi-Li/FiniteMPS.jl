using MKL
using FiniteMPS, FiniteLattices

include("Models/Hubbard.jl")

verbose = 1
GCstep = true
@show Latt = SquaLatt(8, 4, SnakePath)
D = 2000
Para = (t=1, t′=-0.2, U=8, μ = 2)
Ndop = 0 

lsβ = vcat(2.0 .^ (-15:2:-1), 1:8)
lsF = zeros(length(lsβ))
lsE = zeros(length(lsβ))

let

     H = AutomataMPO(U₁SU₂Hubbard(Latt; Para...))

     # ============= SETTN ===============
     ρ, lsF_SETTN = SETTN(H, lsβ[1];
          maxorder=4, verbose=1,
          GCstep=false, GCsweep=true, tol=1e-8,
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

          # TDVPSweep2!(Env, -dβ / 2;
          #      GCstep=GCstep, GCsweep=true, verbose=verbose,
          #      trunc=truncdim(D) & truncbelow(1e-16))

          TDVPSweep1!(Env, -dβ / 2;
               CBEAlg = CheapCBE(D + div(D, 10), 1e-8),
               GCstep=GCstep, GCsweep=true, verbose=verbose,
               trunc=truncdim(D) & truncbelow(1e-16))


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



