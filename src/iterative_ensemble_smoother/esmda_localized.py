"""
Localized ESMDA
---------------

This module implements localized ESMDA, following the paper:
    
    - "Analysis of the performance of ensemble-based assimilation of production and seismic data"
      Alexandre A. Emerick
      
      
API design
----------

The interface uses human-readable names, just like ESMDA.
The implementation (local variables in methods) follows the notation in the paper.

The central idea behind localization is to study equation (B.6) and the Kalman gain K:
    
    M + (\delta M)(\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1) (D_obs - D) =
    M + K (D_obs - D)
    
The Kalman gain K has shape (parameters, observations) and encodes how much each
observation is influenced by each parameter in every update step of the algorithm.
Given a localization matrix of the same shape with entries in the range [0, 1], 
we can "regularize" the kalmain gain:
    
    K_regularized = K * localization    (elementwise product)
    
    
An API might look like:
    
# Create smoohter instance. Set up all global state used in all iterations
smoother = LocalizedESMDA(covariance, observations, alpha, seed, inversion)
X = np.random.randn(...)

for iteration in range(num_assimilations):
    
    # Run simulation and keep track of living indices
    Y, living_idx = g(X)
    
    # Set up all state used for this iteration: 
    # (\delta D)^T [(\delta D) (\delta D)^T + \alpha (N_e - 1) C_D]^(-1) (D_obs - D)
    # With the possibility of dropping dead ensemble members
    
    smoother.prepare_assimilation(Y, living_idx, truncation=0.99)
    
    # Loop over parameter blocks and update them
    for param_block_idx in parameter_indicies_generator():
        
        def localization_callback(K):
            # Logic that ties each parameter index in this block to the observations
            return K * localization
        
        X[param_block_idx, living_idx] = smoother.assimilate(X=X[param_block_idx, living_idx],
                                                             localization_callback=localization_callback
                                                             )
        

Comments
--------
    
- If `localization_callback` is the identity function, LocalizedESMDA is identical to ESMDA.
- The inner loop over parameter blocks saves memory. The result should be the same over any
  possible sequence of parameter blocks.
- The caller is responsible for keeping track of relationships between input parameters and
  observations. For instance, if some points in an input parameter grid are known to be close
  to an observation, the user can create a helper class Grid to keep track of this.

"""

