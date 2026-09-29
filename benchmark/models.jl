module BenchmarkModels

using FiniteMPS

export MODELS, DIMENSIONS, SEED, local_space, hamiltonian, model_parameters

const SEED = 20260929
const DIMENSIONS = (two_site=(128, 256, 512), cbe=(256, 512, 1024))
const MODELS = (
    (id="hubbard_u1u1", name="Hubbard / U1U1", symmetry="U1U1", hubbard=true),
    (id="hubbard_u1su2", name="Hubbard / U1SU2", symmetry="U1SU2", hubbard=true),
    (id="hubbard_z2su2", name="Hubbard / Z2SU2", symmetry="Z2SU2", hubbard=true),
    (id="tj_u1su2", name="t-t′-J-J′ / U1SU2", symmetry="U1SU2", hubbard=false),
)

local_space(model) = !model.hubbard ? U1SU2tJFermion :
    model.symmetry == "U1U1" ? U1U1Fermion :
    model.symmetry == "U1SU2" ? U1SU2Fermion : Z2SU2Fermion

# YC4×8: open x, periodic y, with alternating traversal of each column.
site(x, y) = 4(x - 1) + (isodd(x) ? mod1(y, 4) : 5 - mod1(y, 4))
function lattice_bonds()
    nearest, diagonal = Tuple{Int,Int}[], Tuple{Int,Int}[]
    for x in 1:8, y in 1:4
        push!(nearest, minmax(site(x,y), site(x,y+1)))
        if x < 8
            push!(nearest, minmax(site(x,y), site(x+1,y)))
            for dy in (-1,1)
                push!(diagonal, minmax(site(x,y), site(x+1,y+dy)))
            end
        end
    end
    return sort!(unique!(nearest)), sort!(unique!(diagonal))
end

function model_parameters(model, kind)
    gce = kind == "thermal" || model.symmetry == "Z2SU2"
    params = Dict{String,Any}("lattice"=>"YC4x8", "sites"=>32,
        "boundary"=>"open x, periodic y", "ordering"=>"snake", "t"=>1.0,
        "t_prime"=>-0.2,
        "ensemble"=>gce ? "GCE" : "CE", "mu"=>gce ? 2.0 : 0.0,
        "particles"=>gce ? nothing : 28, "temperature"=>kind == "thermal" ? 1.0 : nothing)
    merge!(params, model.hubbard ? Dict("U"=>8.0) :
        Dict("J"=>0.5,"J_prime"=>0.02,"exchange"=>"J*(Si.Sj-ni*nj/4)"))
    return params
end

function hamiltonian(model, kind)
    F = local_space(model)
    tree = InteractionTree(32)
    nearest, diagonal = lattice_bonds()
    hops = model.symmetry == "U1U1" ?
        ((F.FdagF₊, F.FFdag₊, :up), (F.FdagF₋, F.FFdag₋, :down)) :
        ((F.FdagF, F.FFdag, :spinor),)
    for (bonds, t, J) in ((nearest, 1.0, 0.5), (diagonal, -0.2, 0.02))
        for (i,j) in bonds
            for (creation, annihilation, spin) in hops
                addIntr!(tree, creation, (i,j), (true,true), -t;
                    Z=F.Z, name=(Symbol(:Fdag_,spin), Symbol(:F_,spin)))
                addIntr!(tree, annihilation, (i,j), (true,true), t;
                    Z=F.Z, name=(Symbol(:F_,spin), Symbol(:Fdag_,spin)))
            end
            if !model.hubbard
                addIntr!(tree, F.SS, (i,j), (false,false), J; name=(:S,:S))
                addIntr!(tree, (F.n,F.n), (i,j), (false,false), -J/4; name=(:n,:n))
            end
        end
    end
    mu = model_parameters(model, kind)["mu"]
    for i in 1:32
        model.hubbard && addIntr!(tree, F.nd, i, 8.0; name=:nd)
        iszero(mu) || addIntr!(tree, F.n, i, -mu; name=:n)
    end
    return AutomataMPO(tree)
end

end
