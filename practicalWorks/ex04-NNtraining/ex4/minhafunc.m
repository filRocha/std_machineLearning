function res = minhafunc(x, deriv )
    if deriv == true
        res = x.*(1-x);
    else
        res = 1./(1+exp(-x)); 
    end
end
