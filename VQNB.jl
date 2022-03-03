include("utils.jl")
include("kMeans.jl")

function VQNB(X,y,k)

    (n,d) = size(X)
    
    # Perform K-Means on all the class 1 examples
    # and all the class 2 examples separately
    model1 = kMeans(X[y.==1,:],k)
    model2 = kMeans(X[y.==2,:],k)

    z = zeros(n)
    z[y.==1] = model1.predict(X[y.==1,:])
    z[y.==2] = model2.predict(X[y.==2,:])
    
    m1 = sum(y.==1) # number of times that y^i=1
    m2 = n-m1 # number of times that y^i=2
    p_y = m1/n # proportion of times y^i=1

    p_zy = zeros(k,2) # P(z|y^i)
    for c in 1:k
        p_zy[c,1] = sum(z[y.==1].==c)/m1
        p_zy[c,2] = sum(z[y.==2].==c)/m2
    end

    p_xyz = zeros(d,2,k) # P(x^i_j | z, y^i)
    # For each possible combination of indices j,b,c...
    for j in 1:d
        for b in 1:2
            for c in 1:k
                # Count the number of times that 
                # x^i_j=1, and y^i=1, and z^i=1.
                for i in 1:n
                    if z[i]==c && y[i]==b && X[i,j]==1
                        p_xyz[j,b,c] += 1 
                    end
                end
                # Now that all the values have been 
                # counted up, divide by the number of
                # times that both y^i=1 and z^i=1.
                p_xyz[j,b,c] /= sum(y[z.==c].==b)
            end
        end
    end

    function predict(Xhat)
        
        (t,d) = size(Xhat)
        yhat = zeros(t)
        
        for i in 1:t
            p_yxz = [0.0;0.0]
            for c in 1:k
                # The variables prod_partb (for b=1,2) is the  
                # product over all j of P(x^i_j | z^i=c, y^i=b).
                prod_part1 = 1
                prod_part2 = 1
                for j in 1:d
                    if Xhat[i,j]==1
                        prod_part1 *= p_xyz[j,1,c]
                        prod_part2 *= p_xyz[j,2,c]
                    else
                        prod_part1 *= 1-p_xyz[j,1,c]
                        prod_part2 *= 1-p_xyz[j,2,c]
                    end
                end
                # Now multiply both "prod_parts" by P(z|y^i)
                p_yxz[1] += p_zy[c,1] * prod_part1
                p_yxz[2] += p_zy[c,2] * prod_part2 
            end
            # And finally, multiply by P(y^i)
            p_yxz[1] *= p_y
            p_yxz[2] *= 1-p_y
            
            # Compare the final probabilities and
            # Make a prediction accordingly
            if p_yxz[1] > p_yxz[2]
                yhat[i] = 1
            else
                yhat[i] = 2
            end
        end
        return yhat
    end
    return VQNBModel(predict,p_xyz)
end
