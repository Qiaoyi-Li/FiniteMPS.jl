# julia -t6 --project=$(pwd) main.jl
using FiniteMPS, FiniteLattices

include("model.jl")

verbose = 1
GCstep = false
Latt = YCSqua(8, 4) |> Snake!
D = 512
Para = (t=1, t′=0, μ=-1)

lsβ = vcat(2.0 .^ (-15:2:-1), 1:8)
lsF = zeros(length(lsβ))
lsE = zeros(length(lsβ))

let
     lsF_ex, lsE_ex = ExactSolution(Latt, lsβ; Para...)

     H = AutomataMPO(SpinlessFreeFermion(Latt; Para...))

     # ============= SETTN ===============
     ρ, lsF_SETTN = SETTN(H, lsβ[1];
          maxorder=4, verbose=1,
          GCstep=false, GCsweep=true, tol=1e-12,
          compress = 1e-16,
          trunc=truncdim(256))
     lsF[1] = lsF_SETTN[end]
     lnZ = 2 * log(norm(ρ))
     normalize!(ρ)
     # ==================================

     Env = Environment(ρ', H, ρ)
     canonicalize!(Env, 1)
     lsE[1] = scalar!(Env; normalize=true)
     let δF = (lsF[1] - lsF_ex[1])/abs(lsF_ex[1]), δE = (lsE[1] - lsE_ex[1])/abs(lsE_ex[1])
          println("β = $(lsβ[1]), F = $(lsF[1])(δF/|F| = $(δF)), E = $(lsE[1])(δE/|E| = $(δE))")
          flush(stdout)
     end

     for i = 2:length(lsβ)
          dβ = lsβ[i] - lsβ[i-1]

          TDVPSweep2!(Env, -dβ / 2;
               GCstep=GCstep, GCsweep=true, verbose=verbose,
               trunc=truncdim(D) & truncbelow(1e-16))

          lnZ += 2 * log(norm(ρ))
          normalize!(ρ)

          lsF[i] = -lnZ / lsβ[i]
          lsE[i] = scalar!(Env; normalize=true)

          let δF = (lsF[i] - lsF_ex[i])/abs(lsF_ex[i]), δE = (lsE[i] - lsE_ex[i])/abs(lsE_ex[i])
               println("β = $(lsβ[i]), F = $(lsF[i])(δF/|F| = $(δF)), E = $(lsE[i])(δE/|E| = $(δE))")
               flush(stdout)
          end
     end

end



