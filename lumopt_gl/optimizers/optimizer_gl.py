import numpy as np
import nlopt
from lumopt.optimizers.maximizer import Maximizer

class OptimizerGL(Maximizer):
    """
    Wrapper for the optimizers in NLopt's optimization library.
    This optimizer provides global and hybrid optimization methods that are not present in SciPy.
    It also supports equality and inequality constraints for many of the optimizers.

    Inherits from the optimizer superclass.

    Args:
        target_fom (float): Specifies the target figure of merit (FOM) for the optimization.
        local_method_name (str, optional): Local gradient-based optimization method for hybrid optimizers. 
            Defaults to 'LD_LBFGS'.

    Recommended Methods:
        These methods are tested to provide the best performance for most photonics inverse design tasks.
        Most of these methods require bound constraints.
        Methods that support inequality constraints are marked with an asterisk.

        - Local Optimization: 'LD_LBFGS', 'LD_SLSQP'*, 'LD_MMA'*
        - Hybrid Global Optimization: 'GD_MLSL', 'GD_STOGO'
        - Gradient-Free Optimization: 'GN_AGS'*, 'GN_ORIG_DIRECT_L'*

        For more information, see the NLopt documentation: https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/
    """
    
    def __init__(self, method = "LD_LBFGS", constraints = None, constraints_tol = None, max_iter = None, scaling_factor = 1.0,  ftol = 1.0e-4, scale_initial_gradient_to = 0, penalty_fun = None, penalty_jac = None, target_fom = None, local_method_name = "LD_MMA", min_or_max = 'max'):
        super().__init__(max_iter = max_iter,
                             scaling_factor = scaling_factor,
                             scale_initial_gradient_to = scale_initial_gradient_to,
                             penalty_fun = penalty_fun,
                             penalty_jac = penalty_jac)
        self.method = self._resolve_method(method)
        self.constraints = constraints
        self.constraints_tol = constraints_tol
        if self.constraints and not self.constraints_tol:
            raise ValueError(f"Constraints missing tolerance parameters")
        self.target_fom = target_fom
        self.ftol = ftol
        self.local_method_name = local_method_name
        self.min_or_max = min_or_max
        
    def concurrent_adjoint_solves(self):
        """all local gradient based methods can support concurrent adjoint solves"""
        return self.method in [nlopt.LD_SLSQP, nlopt.LD_MMA, nlopt.LD_LBFGS, nlopt.LD_VAR2, nlopt.LD_VAR1]

    def run(self):
        '''Creates and runs optimizer'''
        opt = self._initialize_nlopt()
        
        print('Running NLopt optimizer')
        print('start = {}'.format(self.start_point))
        
        result = opt.optimize(self.start_point)
        self.result = result
        self.process_nlopt_results(opt, result)
        
    def objective_function(self, x, grad):
        """Objective function that computes both the function value and its gradient."""
        f = self.callable_fom(x)
        if grad.size > 0:
            grad[:] = self.callable_jac(x)
        self.callback(x)
        print(isinstance(f, float))
        return f if isinstance(f, float) else f[0]
    
    def _resolve_method(self, method_name):
        """Resolve string method names to nlopt constants using getattr."""
        try:
            return getattr(nlopt, method_name)
        except AttributeError:
            raise ValueError(f"Unsupported optimization method: {method_name}")
        
    def _initialize_nlopt(self):
        '''Takes stored list of tuples and returns upper and lower bounds as arrays'''
        opt = nlopt.opt(self.method, self.start_point.size)
        if self.min_or_max is 'min':
            opt.set_min_objective(self.objective_function)
        else:
            opt.set_max_objective(self.objective_function)
        opt.set_maxeval(self.max_iter)
        opt.set_ftol_rel(self.ftol)
        if self.target_fom is not None:
            opt.set_stopval(self.target_fom)
        if self.bounds is not None and len(self.bounds) > 0:
            opt.set_lower_bounds([b[0] for b in self.bounds])
            opt.set_upper_bounds([b[1] for b in self.bounds])
        if self.method in [nlopt.GN_MLSL, nlopt.GN_MLSL_LDS, nlopt.GD_MLSL, nlopt.GD_MLSL_LDS]:
            try:
                local_method = getattr(nlopt, self.local_method_name)
                local_opt = nlopt.opt(local_method, self.start_point.size)
                opt.set_local_optimizer(local_opt)
            except AttributeError:
                raise ValueError(f"Unsupported local optimization method: {self.local_method_name}")
        if self.constraints is not None:
                opt.add_inequality_mconstraint(self.constraints,self.constraints_tol)
        return opt
    
    @staticmethod
    def process_nlopt_results(opt,result):
        print("FINAL FOM = ", opt.last_optimum_value())
        print("FINAL PARAMETERS = ", result)
        messages = {
        1: "Convergence achieved ",
        2: (
            "Optimizer stopped because maximum value of criterion function was reached"
        ),
        3: (
            "Optimizer stopped because convergence_relative_criterion_tolerance or "
            "convergence_absolute_criterion_tolerance was reached"
        ),
        4: (
            "Optimizer stopped because convergence_relative_params_tolerance or "
            "convergence_absolute_params_tolerance was reached"
        ),
        5: "Optimizer stopped because max_criterion_evaluations was reached",
        6: "Optimizer stopped because max running time was reached",
        -1: "Optimizer failed",
        -2: "Invalid arguments were passed",
        -3: "Memory error",
        -4: "Halted because roundoff errors limited progress",
        -5: "Halted because of user specified forced stop",
        }
        print(messages[opt.last_optimize_result()])