"""
AEP Numerical Validation Suite
Comprehensive validation of all numerical implementations
Cross-verification, convergence tests, and reproducibility checks
Anti-Entropic Principle Mathematical Foundations
"""

import numpy as np
import matplotlib.pyplot as plt
from proofs.existence_uniqueness import ExistenceUniquenessProof
from proofs.conservation_laws import ConservationLawsProof
from proofs.stability_analysis import StabilityAnalysis
from numerical.parameter_solver import AEPParameterSolver
from numerical.cosmological_integration import CosmologicalIntegration
from validation.error_analysis import ErrorAnalysis

class NumericalValidation:
    """
    Comprehensive numerical validation suite for AEP foundations
    Validates all components work together correctly
    """
    
    def __init__(self):
        self.results = {}
        
    def run_comprehensive_validation(self):
        """
        Run complete validation suite
        """
        print("AEP COMPREHENSIVE NUMERICAL VALIDATION")
        print("=" * 70)
        print("Validating all mathematical foundations implementations...")
        print()
        
        # Test 1: Parameter Solver Validation
        print("TEST 1: PARAMETER SOLVER VALIDATION")
        print("-" * 50)
        param_validation = self.validate_parameter_solver()
        self.results['parameter_solver'] = param_validation
        
        # Test 2: Existence & Uniqueness Validation
        print("\nTEST 2: EXISTENCE & UNIQUENESS VALIDATION")
        print("-" * 50)
        existence_validation = self.validate_existence_uniqueness()
        self.results['existence_uniqueness'] = existence_validation
        
        # Test 3: Conservation Laws Validation
        print("\nTEST 3: CONSERVATION LAWS VALIDATION")
        print("-" * 50)
        conservation_validation = self.validate_conservation_laws()
        self.results['conservation_laws'] = conservation_validation
        
        # Test 4: Stability Analysis Validation
        print("\nTEST 4: STABILITY ANALYSIS VALIDATION")
        print("-" * 50)
        stability_validation = self.validate_stability_analysis()
        self.results['stability_analysis'] = stability_validation
        
        # Test 5: Cosmological Integration Validation
        print("\nTEST 5: COSMOLOGICAL INTEGRATION VALIDATION")
        print("-" * 50)
        integration_validation = self.validate_cosmological_integration()
        self.results['cosmological_integration'] = integration_validation
        
        # Test 6: Error Analysis Validation
        print("\nTEST 6: ERROR ANALYSIS VALIDATION")
        print("-" * 50)
        error_validation = self.validate_error_analysis()
        self.results['error_analysis'] = error_validation
        
        # Test 7: Cross-Module Consistency
        print("\nTEST 7: CROSS-MODULE CONSISTENCY VALIDATION")
        print("-" * 50)
        consistency_validation = self.validate_cross_module_consistency()
        self.results['cross_module_consistency'] = consistency_validation
        
        # Final Summary
        self.print_validation_summary()
        
        return self.results
    
    def validate_parameter_solver(self):
        """Validate parameter solver implementation"""
        print("Validating AEP parameter solver...")
        
        solver = AEPParameterSolver()
        parameters = solver.solve_parameter_system(tolerance=1e-6)
        
        validation_passed = True
        validation_details = {}
        
        # Check convergence
        if parameters['converged']:
            print("✓ Parameter solver converged successfully")
            validation_details['convergence'] = True
        else:
            print("✗ Parameter solver failed to converge")
            validation_details['convergence'] = False
            validation_passed = False
        
        # Check parameter values against expected ranges
        expected_ranges = {
            'g': (1e-4, 1e-2),
            'lambda': (1e-6, 1e-4),
            'kappa': (1e-5, 1e-3),
            'v_chi': (1e-30, 1e-28),
            'lambda_chi': (1e-12, 1e-10),
            'gamma': (1e-3, 1e-1)
        }
        
        print("Parameter value validation:")
        for param, (min_val, max_val) in expected_ranges.items():
            value = parameters[param]
            if min_val <= abs(value) <= max_val:
                print(f"  ✓ {param}: {value:.3e} (within expected range)")
                validation_details[param] = True
            else:
                print(f"  ✗ {param}: {value:.3e} (outside expected range [{min_val:.1e}, {max_val:.1e}])")
                validation_details[param] = False
                validation_passed = False
        
        # Verify AEP relations
        X_min_calculated = parameters['X_min']
        X_min_expected = -1/(8*parameters['g'])
        X_min_error = abs(X_min_calculated - X_min_expected) / abs(X_min_expected)
        
        lambda_calculated = parameters['lambda']
        lambda_expected = (10/np.pi) * parameters['g']**2
        lambda_error = abs(lambda_calculated - lambda_expected) / lambda_expected
        
        print("AEP relation validation:")
        print(f"  X_min relation error: {X_min_error:.2e}")
        print(f"  λ relation error: {lambda_error:.2e}")
        
        if X_min_error < 1e-6 and lambda_error < 1e-6:
            print("✓ AEP relations satisfied")
            validation_details['aep_relations'] = True
        else:
            print("✗ AEP relation errors too large")
            validation_details['aep_relations'] = False
            validation_passed = False
        
        validation_details['passed'] = validation_passed
        return validation_details
    
    def validate_existence_uniqueness(self):
        """Validate existence and uniqueness proofs"""
        print("Validating existence and uniqueness proofs...")
        
        proof = ExistenceUniquenessProof()
        results = proof.theorem_2_proof()
        
        validation_passed = True
        validation_details = {}
        
        # Check Jacobian determinant
        det_jacobian = results['jacobian_determinant']
        if abs(det_jacobian) > 1e-10:
            print(f"✓ Jacobian non-singular: det(J) = {det_jacobian:.2e}")
            validation_details['jacobian'] = True
        else:
            print(f"✗ Jacobian may be singular: det(J) = {det_jacobian:.2e}")
            validation_details['jacobian'] = False
            validation_passed = False
        
        # Check physical constraints
        constraints = results['physical_constraints']
        print("Physical constraint validation:")
        for constraint, satisfied in constraints.items():
            if satisfied:
                print(f"  ✓ {constraint}")
                validation_details[constraint] = True
            else:
                print(f"  ✗ {constraint}")
                validation_details[constraint] = False
                validation_passed = False
        
        validation_details['passed'] = validation_passed
        return validation_details
    
    def validate_conservation_laws(self):
        """Validate energy-momentum conservation"""
        print("Validating conservation laws...")
        
        proof = ConservationLawsProof()
        
        validation_passed = True
        validation_details = {}
        
        # Run conservation verification
        solution = proof.numerical_verification()
        
        if solution.success:
            print("✓ Cosmological evolution computed successfully")
            validation_details['evolution'] = True
            
            # Check conservation violation
            max_violation = proof.check_conservation(solution)
            print(f"Maximum conservation violation: {max_violation:.2e}")
            
            if max_violation < 1e-6:
                print("✓ Energy-momentum conservation verified")
                validation_details['conservation'] = True
            else:
                print("✗ Significant conservation violation detected")
                validation_details['conservation'] = False
                validation_passed = False
        else:
            print("✗ Failed to compute cosmological evolution")
            validation_details['evolution'] = False
            validation_passed = False
        
        validation_details['passed'] = validation_passed
        return validation_details
    
    def validate_stability_analysis(self):
        """Validate stability analysis"""
        print("Validating stability analysis...")
        
        analysis = StabilityAnalysis()
        
        # Theorem 4 validation
        theorem4_results = analysis.theorem_4_proof()
        
        validation_passed = True
        validation_details = {}
        
        print("Stability condition validation:")
        for condition, satisfied in theorem4_results.items():
            if satisfied:
                print(f"  ✓ {condition}")
                validation_details[condition] = True
            else:
                print(f"  ✗ {condition}")
                validation_details[condition] = False
                validation_passed = False
        
        # Theorem 5 validation
        theorem5_results = analysis.theorem_5_proof()
        
        print("Perturbation stability validation:")
        for condition, satisfied in theorem5_results.items():
            if satisfied:
                print(f"  ✓ {condition}")
                validation_details[f"perturbation_{condition}"] = True
            else:
                print(f"  ✗ {condition}")
                validation_details[f"perturbation_{condition}"] = False
                validation_passed = False
        
        validation_details['passed'] = validation_passed
        return validation_details
    
    def validate_cosmological_integration(self):
        """Validate cosmological integration"""
        print("Validating cosmological integration...")
        
        cosmos = CosmologicalIntegration()
        
        validation_passed = True
        validation_details = {}
        
        # Run convergence test
        errors = cosmos.convergence_test()
        
        if len(errors) >= 3:
            convergence_rate = np.log(errors[0]/errors[2]) / np.log(4)  # Since we halved step size twice
            print(f"Measured convergence rate: {convergence_rate:.3f}")
            
            if 3.5 <= convergence_rate <= 4.5:  # Allow some tolerance
                print("✓ O(h⁴) convergence verified")
                validation_details['convergence'] = True
            else:
                print("✗ Convergence rate deviation detected")
                validation_details['convergence'] = False
                validation_passed = False
        else:
            print("✗ Insufficient data for convergence test")
            validation_details['convergence'] = False
            validation_passed = False
        
        # Run cosmological evolution
        try:
            t, y = cosmos.run_cosmological_evolution(z_max=100)
            final_H = cosmos.compute_hubble_parameter(y[:, -1]) * cosmos.Mpc_to_m / 1000
            
            print(f"Final Hubble constant: H₀ = {final_H:.2f} km/s/Mpc")
            
            if 70 <= final_H <= 80:  # Reasonable range
                print("✓ Hubble constant in expected range")
                validation_details['hubble'] = True
            else:
                print("✗ Hubble constant outside expected range")
                validation_details['hubble'] = False
                validation_passed = False
                
        except Exception as e:
            print(f"✗ Cosmological integration failed: {e}")
            validation_details['hubble'] = False
            validation_passed = False
        
        validation_details['passed'] = validation_passed
        return validation_details
    
    def validate_error_analysis(self):
        """Validate error analysis"""
        print("Validating error analysis...")
        
        analysis = ErrorAnalysis()
        
        validation_passed = True
        validation_details = {}
        
        # Run error propagation
        uncertainties = analysis.theorem_8_proof()
        
        # Check uncertainty ranges
        H0_uncertainty = uncertainties['H0_total']
        S8_uncertainty = uncertainties['S8_total']
        
        print("Uncertainty validation:")
        print(f"  H₀ uncertainty: {H0_uncertainty:.2f} km/s/Mpc")
        print(f"  S₈ uncertainty: {S8_uncertainty:.4f}")
        
        if 0.1 <= H0_uncertainty <= 0.5 and 0.001 <= S8_uncertainty <= 0.01:
            print("✓ Uncertainties in expected ranges")
            validation_details['uncertainties'] = True
        else:
            print("✗ Uncertainties outside expected ranges")
            validation_details['uncertainties'] = False
            validation_passed = False
        
        # Run Monte Carlo verification
        try:
            H0_samples, S8_samples = analysis.monte_carlo_verification(n_samples=100)
            H0_std = np.std(H0_samples)
            S8_std = np.std(S8_samples)
            
            print(f"Monte Carlo H₀ std: {H0_std:.2f} km/s/Mpc")
            print(f"Monte Carlo S₈ std: {S8_std:.4f}")
            
            if abs(H0_std - H0_uncertainty) / H0_uncertainty < 0.5:  # Within 50%
                print("✓ Monte Carlo validation consistent")
                validation_details['monte_carlo'] = True
            else:
                print("✗ Monte Carlo validation inconsistent")
                validation_details['monte_carlo'] = False
                validation_passed = False
                
        except Exception as e:
            print(f"✗ Monte Carlo verification failed: {e}")
            validation_details['monte_carlo'] = False
            validation_passed = False
        
        validation_details['passed'] = validation_passed
        return validation_details
    
    def validate_cross_module_consistency(self):
        """Validate consistency between all modules"""
        print("Validating cross-module consistency...")
        
        validation_passed = True
        validation_details = {}
        
        # Test 1: Parameter consistency
        print("Parameter consistency check:")
        solver = AEPParameterSolver()
        params_solver = solver.solve_parameter_system(tolerance=1e-6)
        
        # Compare with expected values from paper
        paper_values = {
            'g': 2.103e-3,
            'lambda': 1.397e-5,
            'kappa': 1.997e-4,
            'v_chi': 1.002e-29,
        }
        
        for param, paper_val in paper_values.items():
            solver_val = params_solver[param]
            relative_error = abs(solver_val - paper_val) / paper_val
            
            if relative_error < 0.01:  # 1% tolerance
                print(f"  ✓ {param}: error = {relative_error:.2%}")
                validation_details[f"param_{param}"] = True
            else:
                print(f"  ✗ {param}: error = {relative_error:.2%}")
                validation_details[f"param_{param}"] = False
                validation_passed = False
        
        # Test 2: Hubble constant consistency
        print("Hubble constant consistency check:")
        cosmos = CosmologicalIntegration()
        t, y = cosmos.run_cosmological_evolution(z_max=10)  # Shorter run for speed
        H0_calc = cosmos.compute_hubble_parameter(y[:, -1]) * cosmos.Mpc_to_m / 1000
        
        expected_H0 = 73.63
        H0_error = abs(H0_calc - expected_H0) / expected_H0
        
        if H0_error < 0.05:  # 5% tolerance
            print(f"  ✓ H₀: calculated = {H0_calc:.2f}, error = {H0_error:.2%}")
            validation_details['hubble_consistency'] = True
        else:
            print(f"  ✗ H₀: calculated = {H0_calc:.2f}, error = {H0_error:.2%}")
            validation_details['hubble_consistency'] = False
            validation_passed = False
        
        # Test 3: Stability consistency
        print("Stability consistency check:")
        stability = StabilityAnalysis()
        ghost_free = stability.analyze_ghost_freedom()
        gradient_stable = stability.analyze_gradient_stability()
        causal = stability.verify_causality()
        
        if ghost_free and gradient_stable and causal:
            print("  ✓ All stability conditions consistent")
            validation_details['stability_consistency'] = True
        else:
            print("  ✗ Stability conditions inconsistent")
            validation_details['stability_consistency'] = False
            validation_passed = False
        
        validation_details['passed'] = validation_passed
        return validation_details
    
    def print_validation_summary(self):
        """Print comprehensive validation summary"""
        print("\n" + "=" * 70)
        print("COMPREHENSIVE VALIDATION SUMMARY")
        print("=" * 70)
        
        total_tests = 0
        passed_tests = 0
        failed_modules = []
        
        for module, results in self.results.items():
            total_tests += 1
            if results.get('passed', False):
                passed_tests += 1
                print(f"✓ {module:30} PASSED")
            else:
                failed_modules.append(module)
                print(f"✗ {module:30} FAILED")
        
        print("-" * 70)
        print(f"Overall: {passed_tests}/{total_tests} modules passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL VALIDATIONS PASSED! 🎉")
            print("AEP mathematical foundations are numerically robust and consistent.")
        else:
            print(f"❌ {len(failed_modules)} modules failed: {', '.join(failed_modules)}")
            print("Please check the implementation of failed modules.")
        
        print("\nVALIDATION DETAILS:")
        for module, results in self.results.items():
            print(f"\n{module}:")
            for test, passed in results.items():
                if test != 'passed':
                    status = "PASS" if passed else "FAIL"
                    print(f"  {test:30} {status}")
    
    def generate_validation_report(self):
        """Generate a detailed validation report"""
        report = {
            'timestamp': np.datetime64('now'),
            'validation_results': self.results,
            'summary': {
                'total_modules': len(self.results),
                'passed_modules': sum(1 for r in self.results.values() if r.get('passed', False)),
                'failed_modules': [module for module, results in self.results.items() if not results.get('passed', True)]
            }
        }
        
        return report

def main():
    """Run complete numerical validation"""
    validator = NumericalValidation()
    
    # Run all validations
    results = validator.run_comprehensive_validation()
    
    # Generate report
    report = validator.generate_validation_report()
    
    print("\n" + "=" * 70)
    print("VALIDATION REPORT GENERATED")
    print("=" * 70)
    print(f"Total modules validated: {report['summary']['total_modules']}")
    print(f"Modules passed: {report['summary']['passed_modules']}")
    print(f"Modules failed: {len(report['summary']['failed_modules'])}")
    
    if report['summary']['passed_modules'] == report['summary']['total_modules']:
        print("\n✅ AEP MATHEMATICAL FOUNDATIONS VALIDATION: SUCCESS")
        print("All implementations are numerically robust and consistent.")
        print("Ready for scientific publication and further research.")
    else:
        print("\n❌ AEP MATHEMATICAL FOUNDATIONS VALIDATION: INCOMPLETE")
        print("Some modules require attention before full deployment.")

if __name__ == "__main__":
    main()
