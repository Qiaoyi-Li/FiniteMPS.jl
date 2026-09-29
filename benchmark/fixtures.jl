module PerformanceFixtures

using FiniteMPS, Random, JSON3
using ..BenchmarkModels

export build_fixture, read_preset

read_preset(model, kind, D) = JSON3.read(read(joinpath(@__DIR__, "presets",
    "$(model.id)_$(kind)_D$(D).json"), String), Dict{String,Any})

function bond_space(symmetry, rows)
    if symmetry == "U1U1"
        return Rep[U₁×U₁]((r[1]//2, r[2]//2)=>r[3] for r in rows)
    elseif symmetry == "U1SU2"
        return Rep[U₁×SU₂]((r[1]//2, r[2]//2)=>r[3] for r in rows)
    else
        return Rep[ℤ₂×SU₂]((r[1], r[2]//2)=>r[3] for r in rows)
    end
end

function build_fixture(rng, model, kind, D)
    preset = read_preset(model, kind, D)
    spaces = [bond_space(model.symmetry, rows) for rows in preset["bonds"]]
    p = local_space(model).pspace
    tensors = [randn(rng, Float64, spaces[i]⊗p,
        kind == "ground" ? spaces[i+1] : p⊗spaces[i+1]) for i in 1:32]
    state = kind == "ground" ? MPS(tensors) : MPO(tensors)
    canonicalize!(state, 32)
    canonicalize!(state, 1)
    normalize!(state)
    H = hamiltonian(model, kind)
    env = Environment(state', H, state)
    canonicalize!(env, 1)
    return (;env, parameters=Dict("bond_dimensions"=>preset["bond_dimensions"],
        "sector_preset"=>"presets/$(model.id)_$(kind)_D$(D).json"))
end

end
