using Pkg

Pkg.activate(".")
Pkg.add(url = "https://github.com/Qiaoyi-Li/FiniteLattices.jl.git", rev="dev")
Pkg.add(url = "https://github.com/Qiaoyi-Li/FiniteMPS.jl.git", rev="dev")

Pkg.resolve()
Pkg.gc()