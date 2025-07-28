using Documenter, FiniteMPS

makedocs(;
     modules = [FiniteMPS],
     sitename="FiniteMPS.jl",
     authors = "Qiaoyi Li")

deploydocs(
    repo = "github.com/Qiaoyi-Li/FiniteMPS.jl",
    devbranch="main",
)