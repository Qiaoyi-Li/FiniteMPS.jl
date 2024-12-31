"""
	 addIntr4!(Root::InteractionTreeNode,
		  Op::NTuple{4,AbstractTensorMap},
		  si::NTuple{4,Int64},
		  strength::Number;
		  Obs::Bool = false,
		  Z::Union{Nothing,AbstractTensorMap} = nothing,
		  name::NTuple{4,Union{Symbol,String}} = (:A, :B, :C, :D)) -> nothing

	 addIntr4!(Tree::InteractionTree, args...) = addIntr4!(Tree.Root.children[1], args...)

Add a 4-site interaction `Op` at site `si` (4tuple) to a given interaction tree. If Z is given, assume each operator in tuple `Op` is fermionic operator and add Z automatically.

	 addIntr4!(Root::InteractionTreeNode,
		  A::LocalOperator,
		  B::LocalOperator,
		  C::LocalOperator,
		  D::LocalOperator,
		  strength::Number,
		  Z::Union{Nothing,AbstractTensorMap};
		  value = nothing) -> nothing

Expert version, each method finally reduces to this one. 

Note if there exist repeated si, it will recurse to `addIntr2!` or `addIntr3!`(TODO) automatically.

# Deprecated
This function is deprecated, use `addIntr!` instead.
"""
function addIntr4!(Root::InteractionTreeNode, O::NTuple{4, AbstractTensorMap}, si::NTuple{4, Int64}, strength::Number;
	Obs::Bool = false,
	Z::Union{Nothing, AbstractTensorMap} = nothing,
	name::NTuple{4, Union{Symbol, String}} = (:A, :B, :C, :D),
	value = Obs ? si => prod(string.(name)) : nothing)
     # support old usage
     Zflag = !isnothing(Z)
	@warn "This function is deprecated, use addIntr! instead"
     return addIntr!(Root, O, si, (Zflag, Zflag, Zflag, Zflag), strength; Obs = Obs, Z = Z, name = name, value = value)
end