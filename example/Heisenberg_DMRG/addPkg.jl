using Pkg

Pkg.activate(".")
Pkg.add(url = "https://github.com/Qiaoyi-Li/FiniteLattices.jl.git", rev="dev")
Pkg.add("FiniteMPS")

Pkg.resolve()
Pkg.gc()