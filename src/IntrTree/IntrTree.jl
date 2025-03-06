mutable struct InteractionChannel{T}
	Ops::Vector{Int64}
	ref::Ref
	LeafL::T
	LeafR::T
	preserve::Bool
	function InteractionChannel(Ops::Vector{Int64}, ref::Ref, LeafL::T, LeafR::T, preserve::Bool = false) where T
		return new{T}(Ops, ref, LeafL, LeafR, preserve)
	end
end
function show(io::IO, obj::InteractionChannel)
	print(io, "InteractionChannel$(obj.Ops)($(obj.ref))")
	return nothing
end

mutable struct InteractionTreeNode
	Op::NTuple{2, Int64} # si, idx
	parent::Union{Nothing, InteractionTreeNode}
	children::Vector{InteractionTreeNode}
	Intrs::Vector{InteractionChannel}
	function InteractionTreeNode(Op::NTuple{2, Int64},
		parent::Union{Nothing, InteractionTreeNode},
		children::Vector{InteractionTreeNode} = InteractionTreeNode[],
		Intrs::Vector{InteractionChannel} = InteractionChannel[])
		return new(Op, parent, children, Intrs)
	end
end
function show(io::IO, obj::InteractionTreeNode)
	print(io, obj.Op[2])
	return nothing
end


parent(node::InteractionTreeNode) = node.parent
children(node::InteractionTreeNode) = node.children
ParentLinks(::Type{InteractionTreeNode}) = StoredParents()
ChildIndexing(::Type{InteractionTreeNode}) = IndexedChildren()
NodeType(::Type{InteractionTreeNode}) = HasNodeType()
nodetype(::Type{InteractionTreeNode}) = InteractionTreeNode

mutable struct InteractionTree{L}
	Ops::Vector{Vector{AbstractLocalOperator}} # Ops[si][idx]
	Refs::Dict{String, Dict}
	RootL::InteractionTreeNode
	RootR::InteractionTreeNode
	function InteractionTree(L::Int64)
		Ops = [AbstractLocalOperator[] for _ in 1:L]
		Refs = Dict{String, Dict}()
		RootL = InteractionTreeNode((0, 0), nothing)
		RootR = InteractionTreeNode((L + 1, 0), nothing)
		return new{L}(Ops, Refs, RootL, RootR)
	end
end
function show(io::IO, obj::InteractionTree{L}) where L
	println(io, typeof(obj), "(")
	for i in 1:L
		print(io, "[")
		for j in 1:length(obj.Ops[i])
			print(io, obj.Ops[i][j])
			j < length(obj.Ops[i]) && print(io, ", ")
		end
		println(io, "]")
	end
	print_tree(io, obj.RootL)
	print_tree(io, obj.RootR)
	print(io, ")")
	return nothing
end


function merge!(Tree::InteractionTree{L}) where L

	lsnode_L = InteractionTreeNode[Tree.RootL]
	lschild_L = InteractionTreeNode[]
	lsnode_R = InteractionTreeNode[Tree.RootR]
	lschild_R = InteractionTreeNode[]
	for si_count in 0:L
		# left to right
		si = si_count
		empty!(lschild_L)
		for node in lsnode_L
			if length(node.Intrs) < 2
				continue
			end

			# search 
			i = 1
			while i < length(node.Intrs)
				ch_i = node.Intrs[i]
				if length(ch_i.Ops) == 0
					ids = findall(1:length(node.Intrs)) do j
						j ≤ i && return false
						ch_j = node.Intrs[j]
						ch_i.preserve || ch_j.preserve && return false
						length(ch_j.Ops) != 0 && return false
						return ch_i.LeafR === ch_j.LeafR
					end

					for j in ids
						ch_j = node.Intrs[j]
						# merge strength
						ch_i.ref[] += ch_j.ref[]
						ch_j.ref[] *= 0

						# delete channel 
						ids_del = findall(ch_j.LeafR.Intrs) do ch
							ch === ch_j
						end
						deleteat!(ch_j.LeafR.Intrs, ids_del)
					end

					deleteat!(node.Intrs, ids)

					i += 1

				else
					ids = findall(1:length(node.Intrs)) do j
						j < i && return false
						ch_j = node.Intrs[j]
						length(ch_j.Ops) == 0 && return false
						return ch_j.Ops[1] == ch_i.Ops[1]
					end
					if length(ids) < 2
						i += 1
						continue
					end

					# add child 
					node_new = InteractionTreeNode((si + 1, ch_i.Ops[1]), node)
					push!(node.children, node_new)
					for j in ids
						ch_j = node.Intrs[j]
						# update ch_j 
						deleteat!(ch_j.Ops, 1)
						ch_j.LeafL = node_new

						push!(node_new.Intrs, ch_j)
					end

					deleteat!(node.Intrs, ids)

				end

			end

			append!(lschild_L, node.children)
		end

		# next layer
		empty!(lsnode_L)
		append!(lsnode_L, lschild_L)

		# right to left
		si = L - si_count + 1
		empty!(lschild_R)
		for node in lsnode_R
			if length(node.Intrs) < 2
				continue
			end

			# search
			i = 1
			while i < length(node.Intrs)
				ch_i = node.Intrs[i]
				if length(ch_i.Ops) == 0
					ids = findall(1:length(node.Intrs)) do j
						j ≤ i && return false
						ch_j = node.Intrs[j]
						ch_i.preserve || ch_j.preserve && return false
						length(ch_j.Ops) != 0 && return false
						return ch_i.LeafL === ch_j.LeafL
					end

					for j in ids
						ch_j = node.Intrs[j]
						# merge strength
						ch_i.ref[] += ch_j.ref[]
						ch_j.ref[] *= 0

						# delete channel
						ids_del = findall(ch_j.LeafL.Intrs) do ch
							ch === ch_j
						end
						deleteat!(ch_j.LeafL.Intrs, ids_del)
					end

					deleteat!(node.Intrs, ids)

					i += 1

				else
					ids = findall(1:length(node.Intrs)) do j
						j < i && return false
						ch_j = node.Intrs[j]
						length(ch_j.Ops) == 0 && return false
						return ch_j.Ops[end] == ch_i.Ops[end]
					end
					if length(ids) < 2
						i += 1
						continue
					end

					# add child
					node_new = InteractionTreeNode((si - 1, ch_i.Ops[end]), node)
					push!(node.children, node_new)
					for j in ids
						ch_j = node.Intrs[j]
						# update ch_j
						deleteat!(ch_j.Ops, length(ch_j.Ops))
						ch_j.LeafR = node_new

						push!(node_new.Intrs, ch_j)
					end

					deleteat!(node.Intrs, ids)

				end

			end

			append!(lschild_R, node.children)
		end

		# next layer
		empty!(lsnode_R)
		append!(lsnode_R, lschild_R)
	end

	# delete refs with zero strength
	for d in values(Tree.Refs)
		filter!(d) do (_, v)
			v[] != 0
		end
	end

	# try to merge on-site terms 
	for node in StatelessBFS(Tree.RootL)
		isempty(node.Intrs) && continue
		i = 1
		while i < length(node.Intrs)
			ch_i = node.Intrs[i]
			if length(ch_i.Ops) != 1
				i += 1
				continue
			end
			ids = findall(1:length(node.Intrs)) do j
				j ≤ i && return false
				ch_j = node.Intrs[j]
				ch_i.preserve || ch_j.preserve && return false
				length(ch_j.Ops) != 1 && return false
				return ch_i.LeafR === ch_j.LeafR
			end

			# sum
			if !isempty(ids)
				si = node.Op[1] + 1
				idx = ch_i.Ops[1]
				Op = deepcopy(Tree.Ops[si][idx])
				Op.strength[] = ch_i.ref[]
				ch_i.ref[] *= 0
				for j in ids
					idx = node.Intrs[j].Ops[1]
					Op_j = deepcopy(Tree.Ops[si][idx])
					Op_j.strength[] = node.Intrs[j].ref[]
					node.Intrs[j].ref[] *= 0
					Op += Op_j
				end
				Op.strength[] = NaN

				# delete
				for idx in vcat(i, ids)
					ch = node.Intrs[idx]
					ids_del = findall(ch.LeafR.Intrs) do ch_j
						ch_j === ch
					end
					deleteat!(ch.LeafR.Intrs, ids_del)
				end
				deleteat!(node.Intrs, vcat(i, ids))


				# add 
				idx = findfirst(x -> x == Op, Tree.Ops[si])
				if isnothing(idx)
					push!(Tree.Ops[si], Op)
					idx = length(Tree.Ops[si])
				end
				# use temp ref 
				ch = InteractionChannel([idx], Ref{Number}(1.0), node, ch_i.LeafR)
				pushfirst!(node.Intrs, ch)
                    push!(ch_i.LeafR.Intrs, ch)
			end

			i += 1
		end
	end

	# delete refs with zero strength
	for d in values(Tree.Refs)
		filter!(d) do (_, v)
			v[] != 0
		end
	end

     # add nodes  
     lsnode_L = InteractionTreeNode[Tree.RootL]
	lschild_L = InteractionTreeNode[]
	lsnode_R = InteractionTreeNode[Tree.RootR]
	lschild_R = InteractionTreeNode[]
	for si_count in 0:L
		# left to right
          si = si_count
          empty!(lschild_L)
          for node in lsnode_L 
               ids = findall(node.Intrs) do ch
                    length(ch.Ops) ≥ 1
               end

               for i in ids 
                    ch = node.Intrs[i]
                    node_new = InteractionTreeNode((si + 1, ch.Ops[1]), node)
                    push!(node.children, node_new)

                    # update ch
                    deleteat!(ch.Ops, 1)
                    ch.LeafL = node_new

                    push!(node_new.Intrs, ch)
               end
               deleteat!(node.Intrs, ids)

               append!(lschild_L, node.children)
          end

          append!(empty!(lsnode_L), lschild_L)

          # right to left
          si = L - si_count + 1
          empty!(lschild_R)
          for node in lsnode_R
               ids = findall(node.Intrs) do ch
                    length(ch.Ops) ≥ 1
               end

               for i in ids
                    ch = node.Intrs[i]
                    node_new = InteractionTreeNode((si - 1, ch.Ops[end]), node)
                    push!(node.children, node_new)

                    # update ch
                    deleteat!(ch.Ops, length(ch.Ops))
                    ch.LeafR = node_new

                    push!(node_new.Intrs, ch)
               end
               deleteat!(node.Intrs, ids)

               append!(lschild_R, node.children)
          end

          append!(empty!(lsnode_R), lschild_R)

     end

	return Tree
end
