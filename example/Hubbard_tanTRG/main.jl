# julia -t6 --project=$(pwd) main.jl
using MKL
using FiniteMPS, FiniteLattices

include("model.jl")

verbose = 1
GCstep = true
Latt = YCSqua(8, 4) |> Snake!
Para = (t=1, t′=-0.2, U = 8.0, μ=2)

lsβ = vcat(2.0 .^ (-15:2:-1), 1:8)
lsF = zeros(length(lsβ))
lsE = zeros(length(lsβ))

lsD = map(lsβ) do β
     β < 1 && return 500
     D = ceil(β/2; digits = 0) * 1000 |> Int
     min(D, 4000)
end

let
     H = AutomataMPO(U₁SU₂Hubbard(Latt; Para...))

     # ============= SETTN ===============
     ρ, lsF_SETTN = SETTN(H, lsβ[1];
          maxorder=2, verbose=1,
          GCstep=false, GCsweep=true, tol=1e-8,
          compress = 1e-12,
          trunc=truncdim(256),
          maxiter = 4)
     lsF[1] = lsF_SETTN[end]
     lnZ = 2 * log(norm(ρ))
     normalize!(ρ)
     # ==================================

     Env = Environment(ρ', H, ρ)
     canonicalize!(Env, 1)
     lsE[1] = scalar!(Env; normalize=true)
     let 
          println("β = $(lsβ[1]), F = $(lsF[1]), E = $(lsE[1])")
          flush(stdout)
     end

     for i = 2:length(lsβ)
          D = lsD[i]
          dβ = lsβ[i] - lsβ[i-1]

          TDVPSweep1!(Env, -dβ / 2;
               CBEAlg = CheapCBE(D + div(D, 4), 1e-8),
               GCstep=GCstep, GCsweep=true, verbose=verbose,
               trunc=truncdim(D) & truncbelow(1e-12))

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



