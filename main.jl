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

for c in 1:k
    for b in 1:2
        imshow(reshape(p[:,b,c],28,28),"gray")
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


