import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.misc import derivative
from sympy import *
from astropy import units as u
from astropy import constants as c

class Galaxy:
    def __init__(self, data):
        self.data = data

        self.rad = data['Rad'].values
        self.sdbul = data['SDbul'].values

        self.rad_u = [R*u.kpc for R in data["Rad"]] #Cylindrical radius
        self.sdbul_u = [S*c.L_sun / (u.pc)**2 for S in data["SDbul"]] #Surface density

        # Interpolate between the data points to get a function to differentiate
        self.popt, self.pcov = curve_fit(Galaxy.surface_density_buldge, self.rad, self.sdbul)

# ---------------- Rho_bulge ---------------------# 

    def rho_bulge(R, z, rho_0, r_b):
        r = (R**2 + z**2)**0.5 
        return rho_0 * np.exp(-r/r_b)    

    def surface_density_buldge(Rs, rho_0, r_b):
        if hasattr(Rs, '__len__'):
            return [2 * quad(Galaxy.rho_bulge, 0, np.inf, args=(R, rho_0, r_b))[0] for R in Rs]
        else:
            return 2 * quad(Galaxy.rho_bulge, 0, np.inf, args=(Rs, rho_0, r_b))[0]

    def sdbul_interpolated(self):
        return Galaxy.surface_density_buldge(self.rad, self.popt[0]*c.L_sun/ (u.pc)**2, self.popt[1]*u.kpc)


    # Differentiate surface density wrt rbar
    def sdbul_prime(self):
        return derivative(self.sdbul_interpolated,self.rad)

    # Define final surface density function

    def rho_buldge_final(self):
        
        def func(R):
            return self.sdbul_prime(R)/R
        return -1/np.pi * quad(func, 0, np.inf)[0]


