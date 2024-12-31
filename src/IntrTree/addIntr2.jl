"""
	 addIntr2!(Root::InteractionTreeNode,
		  Op::NTuple{2,AbstractTensorMap},
		  si::NTuple{2,Int64},
		  strength::Number;
		  Obs::Bool = false,
		  Z::Union{Nothing,AbstractTensorMap} = nothing,
		  name::NTuple{2,Union{Symbol,String}} = (:A, :B)) -> nothing

	 addIntr2!(Tree::InteractionTree, args...) = addIntr2!(Tree.Root.children[1], args...)

Add a two-site interaction `Op` at site `si` (2tuple) to a given interaction tree. If Z is given, assume Op is fermionic operator and add Z automatically.

	 addIntr2!(Root::InteractionTreeNode,
		  OL::LocalOperator,
		  OR::LocalOperator,
		  strength::Number,
		  Z::Union{Nothing,AbstractTensorMap};
		  value = nothing) -> nothing

Expert version, each method finally reduces to this one. The `value` will be stored in the last node.

Note if `OL.si == OR.si`, it will recurse to `addIntr1!` automatically.

# Deprecated
This function is deprecated, use `addIntr!` instead.
"""
function addIntr2!(Root::InteractionTreeNode, Op::NTuple{2, AbstractTensorMap}, si::NTuple{2, Int64}, strength::Number;
	Obs::Bool = false,
	Z::Union{Nothing, AbstractTensorMap} = nothing,
	name::NTuple{2, Union{Symbol, String}} = (:A, :B),
	value = Obs ? si => prod(string.(name)) : nothing)
     # support old usage
     Zflag = !isnothing(Z)
	@warn "This function is deprecated, use addIntr! instead"
     return addIntr!(Root, Op, si, (Zflag, Zflag), strength; Obs = Obs, Z = Z, name = name, value = value)
end

