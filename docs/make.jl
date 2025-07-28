using Documenter, FiniteMPS

pages = ["Home" => "index.md",
         "Tutorial" => ["tutorial/HeisenbergXXZ.md"],
         "Library" => [],
         "Index" => ["index/index.md"]]


makedocs(;
    modules = [FiniteMPS],
    sitename="FiniteMPS.jl",
    authors = "Qiaoyi Li",
    warnonly=[:missing_docs, :cross_references],
    format=Documenter.HTML(; prettyurls=true, mathengine=MathJax()),
    pages = pages,
    pagesonly = true,
)

deploydocs(
    repo = "github.com/Qiaoyi-Li/FiniteMPS.jl",
    devbranch="main",
)
