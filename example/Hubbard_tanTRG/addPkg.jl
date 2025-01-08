using Pkg

# check current path 
@assert pwd()[end-13:end] == "Hubbard_tanTRG"

Pkg.activate(".")
Pkg.add("MKL")
Pkg.add("FiniteMPS")
Pkg.add(url = "https://github.com/Qiaoyi-Li/FiniteLattices.jl.git", rev="dev")


Pkg.resolve()
Pkg.gc()