import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
from scipy.misc import derivative
from sympy import *


# plt.style.use('ggplot')

def read_dat(filename):
    head = []
    with open(filename, 'r') as f:
        for i in range(3):
            head.append(next(f).strip()[2:].split('\t'))

    frame = pd.read_csv(filename, sep='\t', comment='#', names=head[1])

    distance = float(head[0][0].split()[2])
    distance_units = head[0][0].split()[3]

    units = head[2]

    data = {
        'frame': frame,
        'feature_names': head[1],
        'feature_units': units,
        'distance': distance,
        'distance_units': distance_units
    }

    return data


example_data = read_dat("Rotmass\Rotmass\IC4202_rotmass.dat")
example_df = example_data["frame"]

rad = example_df["Rad"].values #Cylindrical radius
sdbul = example_df["SDbul"].values #Surface density



def rho_bulge(z, R, rho_0, r_b):
    r = (R**2 + z**2)**0.5
    return rho_0 * np.exp(-r/r_b)
    

def surface_density_buldge(Rs, rho_0, r_b):
    if hasattr(Rs, '__len__'):
        return [2 * quad(rho_bulge, 0, np.inf, args=(R, rho_0, r_b))[0] for R in Rs]
    else:
        return 2 * quad(rho_bulge, 0, np.inf, args=(Rs, rho_0, r_b))[0]

# Interpolate between the data points to get a function to differentiate
popt, pcov = curve_fit(surface_density_buldge, rad, sdbul) 

def sdbul_interpolated(R):
    return surface_density_buldge(R, popt[0], popt[1])


# Differentiate surface density wrt rbar
def sdbul_prime(R):
    return derivative(sdbul_interpolated, R)

# Define final surface density function

def sdbul_final(R):
    def func(R):
        return sdbul_prime(R)/R
    return -1/np.pi * quad(func, 0, np.inf)[0]

sdbul_final(rad)
# # Checking the fit
# inter = sdbul_interpolated(rad)
# prime = sdbul_prime(rad)

# p = sns.scatterplot(x=rad, y=sdbul)
# p.set_xlabel("Radius")
# p.set_ylabel("Surface Density")
# sns.lineplot(x=rad, y=inter)
# sns.lineplot(x=rad, y=prime)
# plt.show()


