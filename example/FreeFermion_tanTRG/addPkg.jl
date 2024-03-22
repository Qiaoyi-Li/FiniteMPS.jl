using Pkg

Pkg.activate(".")
Pkg.add("FiniteMPS")
Pkg.add(url = "https://github.com/Qiaoyi-Li/FiniteLattices.jl.git", rev="dev")

Pkg.resolve()
Pkg.gc()