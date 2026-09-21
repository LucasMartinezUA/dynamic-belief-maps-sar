P_detect = sum_c P_prior(c) * (1 - (1-p_d)^n_c)
RMST = sum_c P_prior(c) * sum_{t=0}^{T} q^{sum_{s<=t} k_s(c)}
