import numpy as np
from scipy.special import digamma, polygamma


def newton_raphson_theta_step(data, mu, weights, theta, eps=1e-10, theta_max=1e8):
    """
    Perform ONE iteration of Newton-Raphson to update theta with weighted observations.
    
    The weights correspond to a_i = P(z_i=0|y_i), the probability that observation i 
    comes from the NB component (not zero-inflation).
    
    Parameters:
    -----------
    data : array-like
        Observed count data (y_i)
    mu : float
        Mean parameter of NB distribution
    weights : array-like
        Weights a_i = P(z_i=0|y_i) for each observation
    theta : float
        Current theta (dispersion parameter)
    eps : float
        Small value for numerical stability
    theta_max : float
        Maximum allowed value for theta
    
    Returns:
    --------
    float : Updated theta value
    """
    data = np.asarray(data, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if len(data) != len(weights):
        raise ValueError("Data and weights must have the same length")
    
    # Clip parameters
    theta = float(np.clip(theta, eps, theta_max))
    mu = float(np.clip(mu, eps, None))
    
    # Compute first derivative (weighted)
    # f(theta) = sum_i a_i * [psi(y_i + theta) - psi(theta) - ln(theta + mu) + ln(theta) 
    #                         - theta/(theta + mu) - y_i/(theta + mu) + 1]
    
    psi_y_theta = digamma(data + theta)  # ψ(y_i + θ)
    psi_theta = digamma(theta)            # ψ(θ)
    ln_theta_mu = np.log(theta + mu)      # ln(θ + μ)
    ln_theta = np.log(theta)              # ln(θ)
    
    term1 = psi_y_theta - psi_theta
    term2 = -ln_theta_mu + ln_theta
    term3 = -theta / (theta + mu) - data / (theta + mu) + 1.0
    
    f_theta = np.sum(weights * (term1 + term2 + term3))
    
    # Compute second derivative (weighted)
    # f'(theta) = sum_i a_i * [psi_1(y_i + theta) - psi_1(theta) - 1/(theta + mu) 
    #                          + 1/theta - (mu - y_i)/(mu + theta)^2]
    
    psi1_y_theta = polygamma(1, data + theta)  # ψ_1(y_i + θ)
    psi1_theta = polygamma(1, theta)            # ψ_1(θ)
    
    deriv_term1 = psi1_y_theta - psi1_theta
    deriv_term2 = -1.0 / (theta + mu) + 1.0 / theta
    deriv_term3 = -(mu - data) / ((mu + theta) ** 2)
    
    f_prime_theta = np.sum(weights * (deriv_term1 + deriv_term2 + deriv_term3))
    
    # log-theta Newton step
    g = f_theta
    h = f_prime_theta

    g_eta = g * theta
    h_eta = h * theta**2 + g * theta  # chain rule

    if abs(h_eta) < eps:
        return theta

    eta_new = np.log(theta) - g_eta / h_eta
    theta_new = float(np.clip(np.exp(eta_new), eps, theta_max))
    return theta_new 