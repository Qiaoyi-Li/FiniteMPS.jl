using MKL, FiniteMPS, JLD2, BenchmarkTools, TimerOutputs
import FiniteMPS:_preselect, _rightProj,  _directsum_Ar, _expand_Al

PH, Al, Ar_rc = load("CBE/data.jld2", "PH", "Al", "Ar_rc")

D = 20000
CBE(PH, Al, Ar_rc, StandardCBE(D, 1e-8, SweepL2R()))
show(get_timer("CBE"))
Al_ex, Ar_ex, info = CBE(PH, Al, Ar_rc, CheapCBE(D, 1e-8, SweepL2R()))
show(get_timer("CBE"))

# lsD = [10000, 15000, 20000]
# lsRslt = map(lsD) do D
#      println("D = $D")
#      Al_ex, Ar_ex, _ = CBE(PH, Al, Ar_rc, CheapCBE(D, 1e-8, SweepL2R()))
#      @show correctness1 = norm(Al * Ar_rc - Al_ex * Ar_ex)
#      @show orthogonality1 = let Ar_p = permute(Ar_ex.A, ((1,), (2, 3)))
#           norm(Ar_p*Ar_p' - isometry(codomain(Ar_p), codomain(Ar_p)))
#      end

#      Al_ex, Ar_ex, _ = CBE(PH, Al, Ar_rc, StandardCBE(D, 1e-8, SweepL2R()))
#      @show correctness2 = norm(Al * Ar_rc - Al_ex * Ar_ex)
#      @show orthogonality2 = let Ar_p = permute(Ar_ex.A, ((1,), (2, 3)))
#           norm(Ar_p*Ar_p' - isometry(codomain(Ar_p), codomain(Ar_p)))
#      end
#      return (correctness1, orthogonality1, correctness2, orthogonality2)
# end


