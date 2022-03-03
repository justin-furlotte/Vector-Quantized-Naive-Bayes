mutable struct GenericModel
	predict 
end

mutable struct VQNBModel
	predict
	p_xyz
end

# Euclidean distance between all elements of X1 and X2
function distancesSquared(X1,X2)
	(n,d) = size(X1)
	(t,k) = size(X2)
	@assert(d==k)
	return X1.^2*ones(d,t) .+ ones(n,d)*(X2').^2 .- 2X1*X2'
end