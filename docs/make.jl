using Documenter, FiniteMPS

pages = ["Home" => "index.md",
        "Tutorial" => ["tutorial/HeisenbergXXZ.md"],
        "Library" => ["lib/TensorWrappers.md",
            "lib/MPS.md", 
            "lib/Environment.md",
            "lib/ProjHam.md",
            "lib/IntrTree.md"
        ],
        "Index" => ["index/index.md"]]


makedocs(;
    modules = [FiniteMPS],
    sitename="FiniteMPS.jl",
    authors = "Qiaoyi Li",
    warnonly=[:missing_docs, :cross_references],
    pages = pages,
    pagesonly = true,
)

deploydocs(
    repo = "github.com/Qiaoyi-Li/FiniteMPS.jl",
    devbranch="dev",
)
