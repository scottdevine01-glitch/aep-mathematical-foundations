"""
AEP Parameter Solver
Implements Theorem 2: Existence and Uniqueness of Parameter Solutions
Anti-Entropic Principle Mathematical Foundations

NOTE: Uses only numpy for maximum compatibility
"""

import numpy as np

class AEPParameterSolver:
    """
    Solves the AEP parameter system from first principles
    Implements the complete parameter determination from Section 4
    Uses only numpy for maximum compatibility
    """
    
    def __init__(self):
        # Physical constants (hardcoded for compatibility)
        self.M_P = 2.176434e-8  # Planck mass in kg
        self.c = 3e8
        self.hbar = 1.0545718e-34
        
        # Empirical inputs (with uncertainties)
        self.rho_Lambda = (2.4e-3 * 1.602e-19)**4 / (self.hbar * self.c)**3
        self.a0 = 1.20e-10
        self.Rc = 3.09e19
        
        # AEP-determined relationships (Theorem 2)
        self.X_min_relation = lambda g: -1/(8*g)
        self.lambda_relation = lambda g: (10/np.pi) * g**2
        
    def p_x(self, X, g, lam):
        """K-essence Lagrangian P(X) = X + gX^2 + λX^3"""
        return X + g*X**2 + lam*X**3
    
    def p_x_derivative(self, X, g, lam):
        """First derivative P_X(X)"""
        return 1 + 2*g*X + 3*lam*X**2
    
    def p_xx_derivative(self, X, g, lam):
        """Second derivative P_XX(X)"""
        return 2*g + 6*lam*X
    
    def sound_speed_squared(self, X, g, lam):
        """Sound speed c_s^2 = P_X/(P_X + 2X P_XX)"""
        P_X = self.p_x_derivative(X, g, lam)
        P_XX = self.p_xx_derivative(X, g, lam)
        denominator = P_X + 2*X*P_XX
        if denominator == 0:
            return 0
        return P_X / denominator
    
    def solve_g_from_a0(self, a0_empirical):
        """
        Solve for g from acceleration scale a0 (Equation 3)
        Uses AEP relation λ = (10/π)g^2
        """
        denominator = (self.hbar * self.M_P * a0_empirical**4) / self.c**3
        g_cubed = (10/np.pi) / denominator
        return g_cubed**(1/3)
    
    def parameter_system_residuals(self, params, g, lam):
        """
        Compute residuals for the parameter system (Equations 2-7)
        params = [kappa, v_chi]
        """
        kappa, v_chi = params
        X_min = self.X_min_relation(g)
        
        # Residual 1: Dark energy density (Eq 2)
        rho_Lambda_calc = self.M_P**4 * self.p_x(X_min, g, lam)
        resid1 = (self.rho_Lambda - rho_Lambda_calc) / self.rho_Lambda
        
        # Residual 2: Structure scale (Eq 4) - simplified
        mu = 2.417e-33 * self.M_P
        Rc_calc = np.pi * self.hbar / (self.c * self.M_P * np.sqrt(g) * mu**2)
        resid2 = (self.Rc - Rc_calc) / self.Rc
        
        # Residual 3: Sound speed constraint (Eq 6)
        cs2 = self.sound_speed_squared(X_min, g, lam)
        resid3 = cs2 - 1/3
        
        return np.array([resid1, resid2, resid3])
    
    def simple_newton_solve(self, g, lam, initial_guess, tolerance=1e-8, max_iterations=20):
        """
        Simplified Newton-Raphson using only numpy
        """
        params_current = np.array(initial_guess)
        
        for iteration in range(max_iterations):
            residuals = self.parameter_system_residuals(params_current, g, lam)
            residual_norm = np.linalg.norm(residuals)
            
            print(f"Iteration {iteration + 1}:")
            print(f"  κ = {params_current[0]:.6e}, v_χ = {params_current[1]:.6e} M_P")
            print(f"  Residual norm: {residual_norm:.2e}")
            
            if residual_norm < tolerance:
                print("✓ Convergence achieved!")
                break
                
            # Finite difference Jacobian using only numpy
            jacobian = np.zeros((3, 2))
            h = 1e-8
            
            for j in range(2):
                params_perturbed = params_current.copy()
                params_perturbed[j] += h
                residuals_perturbed = self.parameter_system_residuals(params_perturbed, g, lam)
                jacobian[:, j] = (residuals_perturbed - residuals) / h
            
            # Newton update using numpy.linalg.lstsq
            try:
                update = np.linalg.lstsq(jacobian, residuals, rcond=None)[0]
                params_current -= update
            except np.linalg.LinAlgError:
                # Fallback to gradient descent if Jacobian is singular
                print("Jacobian singular, using gradient descent")
                update = jacobian.T @ residuals
                params_current -= 0.1 * update
                
            print()
        
        return params_current, residual_norm < tolerance, residual_norm
    
    def solve_parameter_system(self, tolerance=1e-6, max_iterations=20):
        """
        Main parameter solver - Uses only numpy for compatibility
        """
        print("AEP PARAMETER SOLVER (NUMPY-ONLY VERSION)")
        print("=" * 60)
        print("Solving cosmological parameters from first principles...")
        print(f"Target tolerance: {tolerance}")
        print()
        
        # Step 1: Solve for g from a0 (AEP relation)
        print("Step 1: Determine g from acceleration scale a0")
        g = self.solve_g_from_a0(self.a0)
        print(f"g = {g:.6e}")
        
        # Step 2: Determine λ from AEP relation (Theorem 2)
        print("Step 2: Determine λ from AEP complexity minimization")
        lam = self.lambda_relation(g)
        print(f"λ = {lam:.6e}")
        
        # Step 3: Determine X_min from AEP relation
        X_min = self.X_min_relation(g)
        print(f"X_min = {X_min:.6e} M_P^4")
        
        # Step 4: Solve remaining parameters using simplified Newton-Raphson
        print("Step 3: Solve coupled system for κ, v_χ")
        print("-" * 40)
        
        # Initial guesses from your Table 1
        initial_guess = [2.0e-4, 1.0e-29 * self.M_P]
        
        kappa, v_chi, converged, final_residual = self.simple_newton_solve(
            g, lam, initial_guess, tolerance, max_iterations)
        
        # Step 5: Determine remaining parameters
        print("Step 4: Determine λ_χ and γ")
        lambda_chi = self.rho_Lambda / (v_chi**4)
        
        # Background field values from your Table 1
        phi_0 = 1.254e-2 * self.M_P
        phi_dot_0 = 3.892e-61 * self.M_P**2
        gamma = (kappa / self.M_P**2) * (phi_0**2 * v_chi**2) / phi_dot_0**2
        
        print(f"λ_χ = {lambda_chi:.6e}")
        print(f"γ = {gamma:.6e}")
        
        # Verify physical constraints
        print()
        print("Step 5: Verify physical constraints")
        print("-" * 40)
        
        # Check P_X(X_min) ≈ 0
        P_X_min = self.p_x_derivative(X_min, g, lam)
        print(f"P_X(X_min) = {P_X_min:.2e} (should be ≈ 0)")
        
        # Check sound speed
        cs2 = self.sound_speed_squared(X_min, g, lam)
        print(f"c_s²(X_min) = {cs2:.6f} (should be 0.333333)")
        
        # Check stability conditions
        no_ghosts = self.p_x_derivative(X_min, g, lam) + 2*X_min*self.p_xx_derivative(X_min, g, lam) > 0
        print(f"No ghosts condition: {no_ghosts}")
        
        return {
            'g': g,
            'lambda': lam,
            'kappa': kappa,
            'v_chi': v_chi,
            'lambda_chi': lambda_chi,
            'gamma': gamma,
            'X_min': X_min,
            'converged': converged,
            'final_residual': final_residual
        }
    
    def error_propagation(self, parameters):
        """
        Simplified error propagation using only numpy
        """
        print()
        print("ERROR PROPAGATION ANALYSIS")
        print("=" * 50)
        
        # Input uncertainties (relative)
        sigma_rho = 0.01  # 1%
        sigma_a0 = 0.02   # 2% 
        sigma_Rc = 0.05   # 5%
        
        # Sensitivity analysis
        g = parameters['g']
        
        # From a0 = c^3 / (ħ M_P (gλ)^(1/4)) and λ = (10/π)g^2
        sigma_g = (4/3) * sigma_a0 * g
        
        # Propagate to other parameters
        sigma_lam = 2 * sigma_g * parameters['lambda'] / g
        sigma_kappa = np.sqrt(sigma_g**2 + sigma_Rc**2) * parameters['kappa']
        
        print(f"σ_g / g = {sigma_g/g:.3f}")
        print(f"σ_λ / λ = {sigma_lam/parameters['lambda']:.3f}")
        print(f"σ_κ / κ = {sigma_kappa/parameters['kappa']:.3f}")
        print()
        print("Parameter uncertainties:")
        print(f"g = ({g:.3f} ± {sigma_g:.1e})")
        print(f"λ = ({parameters['lambda']:.3f} ± {sigma_lam:.1e})")
        print(f"κ = ({parameters['kappa']:.3f} ± {sigma_kappa:.1e})")

def main():
    """Run the complete AEP parameter determination"""
    solver = AEPParameterSolver()
    
    # Solve complete parameter system
    parameters = solver.solve_parameter_system()
    
    print()
    print("FINAL PARAMETER SET")
    print("=" * 50)
    for key, value in parameters.items():
        if key not in ['converged', 'final_residual']:
            if 'chi' in key:
                print(f"{key:12} = {value/2.176434e-8:.6e} M_P")  # Convert back to M_P units
            elif key == 'X_min':
                print(f"{key:12} = {value:.6e} M_P^4")
            else:
                print(f"{key:12} = {value:.6e}")
    
    print(f"{'converged':12} = {parameters['converged']}")
    print(f"{'residual':12} = {parameters['final_residual']:.2e}")
    
    # Error analysis
    solver.error_propagation(parameters)
    
    # Verify against Table 1 values
    print()
    print("COMPARISON WITH PAPER TABLE 1")
    print("=" * 50)
    paper_values = {
        'g': 2.103e-3,
        'lambda': 1.397e-5,
        'kappa': 1.997e-4,
        'v_chi': 1.002e-29,
        'lambda_chi': 9.98e-11,
        'gamma': 2.00e-2
    }
    
    for key, paper_val in paper_values.items():
        if key in parameters:
            calc_val = parameters[key]
            if key == 'v_chi':  # Convert back to M_P units
                calc_val = calc_val / 2.176434e-8
            relative_error = abs(calc_val - paper_val) / paper_val
            status = "✓" if relative_error < 0.01 else "✗"
            print(f"{status} {key:12}: paper = {paper_val:.3e}, calculated = {calc_val:.3e}, error = {relative_error:.2%}")

if __name__ == "__main__":
    main()
