using Documenter, FiniteMPS

pages = ["Home" => "index.md",
	"Tutorial" => ["tutorial/Threading.md",
		"tutorial/Hamiltonian.md",
		"tutorial/Observable.md",
		"tutorial/Heisenberg.md",
		"tutorial/Hubbard.md"],
	"Local Space" => ["localspace/Spin.md",
		"localspace/Fermion.md",
	],
	"Library" => ["lib/TensorWrappers.md",
		"lib/MPS.md",
		"lib/Environment.md",
		"lib/ProjHam.md",
		"lib/IntrTree.md",
		"lib/ObsTree.md",
		"lib/ITP.md",
		"lib/Algebra.md",
		"lib/Algorithm.md",
		"lib/Deprecate.md",
	],
	"Index" => ["index/index.md"],
]


makedocs(;
	modules = [FiniteMPS],
	sitename = "FiniteMPS.jl",
	authors = "Qiaoyi Li",
	warnonly = [:missing_docs, :cross_references],
	pages = pages,
	pagesonly = true,
)


if haskey(ENV, "GITHUB_REF")
	if ENV["GITHUB_REF"] == "main"
		devbranch = "main"
		devurl = "stable"
	elseif ENV["GITHUB_REF"] == "dev"
		devbranch = "dev"
		devurl = "dev"

	end
	deploydocs(
		repo = "github.com/Qiaoyi-Li/FiniteMPS.jl",
		devbranch = devbranch,
		devurl = devurl,
		push_preview = true,
	)
end
