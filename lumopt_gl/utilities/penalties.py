"""
penalties.py

This module contains penalty functions, only for shape optimization for now.
"""
import numpy as np
import autograd.numpy as np
from autograd import grad

def smoothing_penalty(parameters, penalty_scale=1e12):
    """Calculates a penalty based on the squared differences between consecutive parameters."""
    differences = np.diff(parameters)
    squared_differences = differences**2
    total_penalty = penalty_scale * np.sum(squared_differences)
    return total_penalty

def curvature_penalty(parameters):
    """Calculates a penalty based on radius of curvature."""
    def r_fun_2d(dx, dy, ddx, ddy):
        """
        Returns radius of curvature.
        """
        numer = (dx**2 + dy**2)**(3/2)
        denom = dx * ddy - dy * ddx
        return numer / np.abs(denom)
    
    def r_exact(t):
        """
        Exact value of R.
        """
        numer = ((np.cos(t)/(2*np.sqrt(t))-np.sqrt(t)*np.sin(t))**2 + np.cos(t)**2)**(3/2)
        denom = -(np.cos(t)/(2*np.sqrt(t))-np.sqrt(t)*np.sin(t))*np.sin(t) - np.cos(t)*(-np.cos(t)/(4*t**(3/2))-np.sin(t)/np.sqrt(t)-np.sqrt(t)*np.cos(t))
        R = numer / np.abs(denom)
        if t[0] == 0:
            R[0] = 1/2
        return R
    
    def curvature_radius(x, y, h):
        """
        Compute radius of curvature numerically.
        """
        hsquare = h**2
        r = np.zeros(x.size)

        # Forward finite difference
        fwd1 = np.array([-25/12, 4, -3, 4/3, -1/4])
        fwd2 = np.array([15/4, -77/6, 107/6, -13, 61/12, -5/6])

        # Forward finite difference minus 1
        ffd1 = np.array([-1/4, -5/6, 3/2, -1/2, 1/12])
        ffd2 = np.array([5/6, -5/4, -1/3, 7/6, -1/2, 1/12])

        # Central finite difference
        cfd1 = np.array([1/12, -2/3, 0, 2/3, -1/12])
        cfd2 = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])

        # Backward finite difference
        bfd1 = np.array([1/4, -4/3, 3, -4, 25/12])                  # Accuracy 4
        bfd2 = np.array([-5/6, 61/12, -13, 107/6, -77/6, 15/4])     # Accuracy 4

        bfd1p1 = np.array([-1/12, 1/2, -3/2, 5/6, 1/4])             # Accuracy 4
        bfd2p1 = np.array([1/12, -1/2, 7/6, -1/3, -5/4, 5/6])       # Accuracy 4

        dx  = np.dot(fwd1, x[0:5]) / h
        dy  = np.dot(fwd1, y[0:5]) / h
        ddx = np.dot(fwd2, x[0:6]) / hsquare
        ddy = np.dot(fwd2, y[0:6]) / hsquare
        r[0] = r_fun_2d(dx, dy, ddx, ddy)

        dx  = np.dot(ffd1, x[0:5]) / h
        dy  = np.dot(ffd1, y[0:5]) / h
        ddx = np.dot(ffd2, x[0:6]) / hsquare
        ddy = np.dot(ffd2, y[0:6]) / hsquare
        r[1] = r_fun_2d(dx, dy, ddx, ddy)

        for i in range(2, r.size-2):
            # 1st derivative
            # Central finite difference
            # Accuracy 4
            dx  = np.dot(cfd1, x[i-2:i+3]) / h
            dy  = np.dot(cfd1, y[i-2:i+3]) / h

            # 2nd derivative
            # Central finite difference
            # Accuracy 4
            ddx = np.dot(cfd2, x[i-2:i+3]) / hsquare
            ddy = np.dot(cfd2, y[i-2:i+3]) / hsquare

            r[i] = r_fun_2d(dx, dy, ddx, ddy)

        dx  = np.dot(bfd1p1, x[-5:]) / h
        dy  = np.dot(bfd1p1, y[-5:]) / h
        ddx = np.dot(bfd2p1, x[-6:]) / hsquare
        ddy = np.dot(bfd2p1, y[-6:]) / hsquare
        r[-2] = r_fun_2d(dx, dy, ddx, ddy)

        dx  = np.dot(bfd1, x[-5:]) / h
        dy  = np.dot(bfd1, y[-5:]) / h
        ddx = np.dot(bfd2, x[-6:]) / hsquare
        ddy = np.dot(bfd2, y[-6:]) / hsquare
        r[-1] = r_fun_2d(dx, dy, ddx, ddy)

        return r
    
    return curvature_radius

def curvature_penalty_differential(parameters, min_radius=0.150, alpha=1.0, kappa=10.0):
    """Adapted from Tidy3D."""
    def quad_fit(p0, pc, p2):
        p1 = 2 * pc - p0/2 - p2/2

        def p(t):
            return (1-t)**2 * (p0 - p1) + p1 + t**2 * (p2 - p1)

        def d_p(t):
            return 2 * (1-t) * (p1 - p0) + 2 * t * (p2 - p1)

        def d2_p(t):
            return 2 * p0 - 4 * p1 + 2 * p2
        
        return p, d_p, d2_p

    def get_fit_vals(xs, ys):
        ps = np.stack((xs, ys), axis=1)
        p0 = ps[:-2]
        pc = ps[1:-1]
        p2 = ps[2:]

        p, d_p, d_2p = quad_fit(p0, pc, p2)
        ps = p(0.5)
        dps = d_p(0.5)
        d2ps = d_2p(0.5)
        return ps.T, dps.T, d2ps.T

    def get_radii_curvature(xs, ys):
        ps, dps, d2ps = get_fit_vals(xs, ys)
        xp, yp = dps
        xp2, yp2 = d2ps
        num = (xp**2 + yp**2) ** (3.0/2.0)
        den = np.abs(xp * yp2 - yp * xp2) + 1e-2
        return num / den

    def penalty_fn(radius):
        return alpha * (1 / (1 + np.exp(-kappa * (min_radius - radius))))

