"""
    addIntr3!(Root::InteractionTreeNode,
         Op::NTuple{3,AbstractTensorMap},
         si::NTuple{3,Int64},
         strength::Number;
         Obs::Bool = false,
         Z::Union{Nothing,AbstractTensorMap} = nothing,
         fermionic::NTuple{3,Bool} = (false, false, false),
         name::NTuple{3,Union{Symbol,String}} = (:A, :B, :C)) -> nothing

    addIntr3!(Tree::InteractionTree, args...) = addIntr3!(Tree.Root.children[1], args...)

Add a 3-site interaction `Op` at site `si` (3tuple) to a given interaction tree. If Z is given, assume operators in `Op` are fermionic and apply the Z transformation automatically.

    addIntr3!(Root::InteractionTreeNode,
         A::LocalOperator,
         B::LocalOperator,
         C::LocalOperator,
         strength::Number,
         Z::Union{Nothing,AbstractTensorMap};
         fermionic::NTuple{3,Bool}=(false, false, false),
         value = nothing) -> nothing

Expert version, each method finally reduces to this one. The `value` will be stored in the last node.

Note if there exist repeated si, it will recurse to `addIntr2!` or `addIntr1!` automatically.
"""
function addIntr3!(Root::InteractionTreeNode, Op::NTuple{3,AbstractTensorMap}, si::NTuple{3,Int64}, strength::Number;
     Obs::Bool=false,
     Z::Union{Nothing,AbstractTensorMap}=nothing,
     fermionic::NTuple{3,Bool}=(false, false, false),
     name::NTuple{3,Union{Symbol,String}}=(:A, :B, :C))

    # Convert names to strings
    name = string.(name)

    strength == 0 && return nothing
    value = Obs ? (prod(name), si...) : nothing

    (A, B, C) = map(1:3) do i
        LocalOperator(Op[i], name[i], si[i])
    end

    # Ensure site indices are in ascending order
    if si[1] > si[2]
        A, B = _swap(A, B)
        si = (si[2], si[1], si[3])
        fermionic = (fermionic[2], fermionic[1], fermionic[3])
        !isnothing(Z) && fermionic[1] && _addZ!(B, Z)
    end
    if si[2] > si[3]
        B, C = _swap(B, C)
        si = (si[1], si[3], si[2])
        fermionic = (fermionic[1], fermionic[3], fermionic[2])
        !isnothing(Z) && fermionic[2] && _addZ!(C, Z)
    end
    if si[1] > si[2]
        A, B = _swap(A, B)
        si = (si[2], si[1], si[3])
        fermionic = (fermionic[2], fermionic[1], fermionic[3])
        !isnothing(Z) && fermionic[1] && _addZ!(B, Z)
    end

    _addtag!(A, B, C)

    # ============ reduce to 1-site term ===========
    if si[1] == si[2] == si[3]
        return addIntr1!(Root, A * B * C, strength, nothing; value=value)
    # ============ reduce to 2-site term ===========
    elseif si[1] == si[2]
        if fermionic[1] == fermionic[2]
            #         C           C
            # B Z Z Z Z  or  B       -->        C
            # A Z Z Z Z      A            AB
            return addIntr2!(Root, A * B, C, strength, nothing; value=value)
        else
            #         C              C
            # B          or  B Z Z Z Z  -->           C
            # A Z Z Z Z      A               AB Z Z Z Z
        end
    elseif si[2] == si[3]
        if fermionic[1] == true
            #         C
            #         B  -->         BC
            # A Z Z Z Z      A Z Z Z Z
            return addIntr2!(Root, A, B * C, strength, Z; value=value)
        else
            #     C
            #     B  -->       BC
            # A           A
            return addIntr2!(Root, A, B * C, strength, nothing; value=value)
        end
    else
        return addIntr3!(Root, A, B, C, strength, Z; fermionic=fermionic, value=value)
    end

    @assert false # make sure all cases are considered
end

function addIntr3!(Root::InteractionTreeNode,
     A::LocalOperator, B::LocalOperator, C::LocalOperator,
     strength::Number, Z::Union{Nothing,AbstractTensorMap};
     fermionic::NTuple{3,Bool}=(false, false, false), value=nothing)

    if isnothing(Z)
        # note addIntr4! may propagate a true even if Z == nothing
        fermionic = (false, false, false)
    end


    @assert A.si < B.si < C.si
    # only consider even number of fermionic operators
    @assert !(fermionic[1] ⊻ fermionic[2] ⊻ fermionic[3])

    #           C
    #      B [Z Z]
    # A [Z Z  Z Z]

    if !isnothing(Z)
        fermionic[1] && _addZ!(B, Z)
        (fermionic[1] != fermionic[2]) && _addZ!(C, Z)
    end

    current_node = Root
    si = 1
    pspace = getPhysSpace(A)

    while si < C.si
        if si == A.si
            Op_i = A
        elseif si == B.si
            Op_i = B
        elseif !isnothing(Z)
            if fermionic[1] && A.si < si < B.si
                Op_i = LocalOperator(Z, :Z, si)
            elseif (fermionic[1] != fermionic[2]) && B.si < si < C.si
                Op_i = LocalOperator(Z, :Z, si)
            else 
                Op_i = IdentityOperator(pspace, si)
            end
        else
            Op_i = IdentityOperator(pspace, si)
        end
        idx = findfirst(current_node.children) do x
            x.Op ≠ Op_i && return false
            if hastag(x.Op) && hastag(Op_i)
                 x.Op.tag ≠ Op_i.tag && return false
            end
            return true
        end
        if isnothing(idx)
            addchild!(current_node, Op_i)
            current_node = current_node.children[end]
        else
            current_node = current_node.children[idx]
            # replace the tag
            hastag(current_node.Op) && (current_node.Op.tag = Op_i.tag)
        end
        si += 1
    end

    # Add the last operator (C)
    idx = findfirst(x -> x.Op == C, current_node.children)
    if isnothing(idx)
        addchild!(current_node, C, value)
        current_node.children[end].Op.strength = strength
    else
        if !isnothing(value)
            current_node.children[idx].value = value
        end
        _update_strength!(current_node.children[idx], strength) && deleteat!(current_node.children, idx)
    end

    return nothing
end

_addtag!(::LocalOperator{1, 1}, ::LocalOperator{1, 1}, ::LocalOperator{1, 1}) = nothing

function _addtag!(A::LocalOperator{1,2}, B::LocalOperator{2,2}, C::LocalOperator{2,1})
    name = map(x -> x.name, [A, B, C])
    for i = 2:3 # make sure each name is unique
         if any(==(name[i]), view(name, 1:i-1))
              name[i] = name[i] * "$i"
         end
    end
    A.tag = (("phys",), ("phys", "$(name[1])<-$(name[2])"))
    B.tag = (("$(name[1])<-$(name[2])", "phys"), ("phys", "$(name[2])<-$(name[3])"))
    C.tag = (("$(name[2])<-$(name[3])", "phys"), ("phys",))
    return nothing
end
