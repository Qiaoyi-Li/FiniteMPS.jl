"""
	 addIntr1!(Root::InteractionTreeNode,
		  Op::AbstractTensorMap,
		  si::Int64,
		  strength::Number;
		  Obs::Bool = false,
		  name::Union{Symbol,String} = :O) -> nothing

	 addIntr1!(Tree::InteractionTree, args...) = addIntr1!(Tree.Root.children[1], args...)

Add an on-site term `Op` at site `si` to a given interaction tree. 

	 addIntr1!(Root::InteractionTreeNode,
		  O::LocalOperator,
		  strength::Number;
		  value = nothing) -> nothing

Expert version, each method finally reduces to this one. The `value` will be stored in the last node.

# Deprecated
This function is deprecated, use `addIntr!` instead.
"""
function addIntr1!(Root::InteractionTreeNode, Op::AbstractTensorMap, si::Int64, strength::Number;
		Obs::Bool = false,
		name::Union{Symbol, String} = :O,
		value = Obs ? (si,) => string(name) : nothing)
		# convert to string
		name = string(name)
	@warn "This function is deprecated, use addIntr! instead"
	return addIntr!(Root, Op, si, strength; Obs = Obs, name = name, value = value)
end

