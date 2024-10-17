using Pkg

# check current path 
@assert pwd()[end-13:end] == "Hubbard_tanTRG"

Pkg.activate(".")
Pkg.develop(path = "../../")
Pkg.add(url = "https://github.com/Qiaoyi-Li/FiniteLattices.jl.git", rev="dev")
Pkg.add("MKL")

Pkg.resolve()
Pkg.gc()