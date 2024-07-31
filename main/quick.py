import math

if __name__ == "__main__":

    rings = 25
    rays = 20
    # Diffusion coefficient
    d = 1
    # Domain Radius
    r = 1
    # Run time
    t = 1

    d_radius = r / rings
    d_theta = ((2 * math.pi) / rays)
    # d_theta = 0.3306939635357677
    d_time = (0.1 * min(d_radius * d_radius, d_theta * d_theta)) / (2 * d)

    '''
        This factor was chosen due to its relevance to the number of angular rays (the value of N)
        
        The underlying computation using this value of N:    
        
       [ d_time / (m * d_radius * d_theta)  * (J_R_Theta - J_L_Theta)) ] 
       
        J_R_Theta = ( - phi(k,m,n+1) - phi(k,m,n) ) / (m * d_radius * d_theta)
       
        J_L_Theta = ( - phi(k,m,n) - phi(k,m,n-1) / (m * d_radius * d_theta) 
    
    '''

    factor = - d_time * (1/((d_radius * d_theta) ** 2))
    print(f'Factor from the computation of phi: {factor}')

    sum_factor = (d_theta * d_time) * (1 / (math.pi * d_radius))

    print(f'Factor from the sum: {sum_factor}')

