# Import the data
using JLD, Printf, PyPlot
data = load("mnist35.jld")
(X, y, Xtest, ytest) = (data["X"], data["y"], data["Xtest"], data["ytest"])

# All pixels the image values become binary (1 or 2)
X[X.>0.5] .= 2 
X[X.<0.5] .= 1
X = Int64.(X)
Xtest[Xtest.>0.5] .= 2
Xtest[Xtest.<0.5] .= 1
Xtest = Int64.(Xtest)
(n,d) = size(X)

# Fit the Vector-Quantized Naive Bayes model
include("VQNB.jl")
k = 5
model = VQNB(X,y,k)
p = model.p_xyz # See VQNB.jl for description of p_xyz
Xprob = zeros(d,2,k)

# The photos are reflected and rotated 90 degrees.
# This function orients them properly
function Orient(P)
    n = size(P,1)
    # reflect along vertical midpoint
    for i in 1:n
        for j in 1:Int(floor(n/2))
            temp = P[i,j]
            P[i,j] = P[i,n+1-j]
            P[i,n+1-j] = temp
        end
    end

    # rotate counter-clockwise 90 degrees
    # by taking transpose and then
    # reversing every column
    P = transpose(P)
    for j in 1:n
        P[:,j] = P[:,j][end:-1:1]
    end

    return P
end

# The photos are reflected and rotated 90 degrees.
for c in 1:k
    for b in 1:2
        pr = reshape(p[:,b,c],28,28)
        pr = Orient(pr)

        imshow(pr,"gray")
        title(string("z=",c,", y=",b))
        display(gcf())
        savefig(string("y",b,"z",c,".png"))
    end
end

@printf("Predicting...\n")

# Compute test error
yhat = model.predict(Xtest)
E = sum(yhat .!= ytest)/size(Xtest,1)
@printf("Error rate (Vector-Quantized Naive Bayes) = %.2f\n", E)


