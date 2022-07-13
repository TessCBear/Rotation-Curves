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
from astropy.coordinates import Angle


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

rad = example_df["Rad"].values*u.kpc #Cylindrical radius


# ---------------- Rho_bulge ---------------------# 


sdbul = example_df["SDbul"].values*c.L_sun / (u.pc)**2 #Surface density



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
    return surface_density_buldge(R, popt[0]*c.L_sun/ (u.pc)**2, popt[1]*u.kpc)


# Differentiate surface density wrt rbar
def sdbul_prime(R):
    return derivative(sdbul_interpolated, R)

# Define final surface density function

def rho_buldge_final(R):
    
    def func(R):
        return sdbul_prime(R)/R
    return -1/np.pi * quad(func, 0, np.inf)[0]


# Checking the fit
inter = sdbul_interpolated(rad)
prime = sdbul_prime(rad)

p = sns.scatterplot(x=rad, y=sdbul)
p.set_xlabel("Radius")
p.set_ylabel("Surface Density")
sns.lineplot(x=rad, y=inter)
sns.lineplot(x=rad, y=prime)
plt.show()

# ---------------------- Rho_gas ----------------------- #
sdgas = example_df["SDgas"].values *c.L_sun / (u.pc)**2

def surface_density_gas(R, sigma0, c1, c2, c3, c4, R_sig):
    return sigma0*(1 + c1*R + c2*R**2 + c3*R**3 + c4*R**4) * np.exp(-R/R_sig)

qopt, qcov = curve_fit(surface_density_gas, rad, sdgas, maxfev=5000) 

def sdgas_interpolated(R):
    return surface_density_gas(R, qopt[0], qopt[1], qopt[2], qopt[3], qopt[4], qopt[5])

# # Checking the fit

# inter_gas = sdgas_interpolated(rad)
# p = sns.scatterplot(x=rad, y=sdgas)
# p.set_xlabel("Radius")
# p.set_ylabel("Surface Density")
# sns.lineplot(x=rad, y=inter_gas)
# plt.show()

def rho_gas(R):
    return 1.4 * sdgas_interpolated(R)/ (np.sqrt(2*np.pi)* 0.130*u.kpc) # 1.4 to account for lack of H2 data, 0.130 kpc from Hossenfelder et al

#--------------------Rho_disk (*)---------------------------#

sddisk = example_df["SDdisk"].values *c.L_sun / (u.pc)**2

def rho_disk(z, R, rho_0, Rstar):
    zstar = 0.196*Rstar**0.633 *u.kpc # following Hossenfelder
    z*= u.kpc
    ang = z/zstar * u.radian
    return rho_0*np.exp(-R/Rstar)*(1-(np.tanh(ang))**2)

def surface_density_disk(Rs, rho_0, Rstar):
    if hasattr(Rs, '__len__'):
        return [2 * quad(rho_disk, 0, np.inf, args=(R, rho_0, Rstar))[0] for R in Rs]
    else:
        return 2 * quad(rho_disk, 0, np.inf, args=(Rs, rho_0, Rstar))[0]

ropt, rcov = curve_fit(surface_density_disk, rad, sddisk) 

def sddisk_interpolated(R):
    return surface_density_disk(R, ropt[0], ropt[1])

def rho_disk_final(R):
    return sddisk_interpolated(R)/(np.sqrt(2*np.pi)*0.196*ropt[1]**0.633)

# # Checking the fit
# inter = sddisk_interpolated(rad)

# p = sns.scatterplot(x=rad, y=sddisk)
# p.set_xlabel("Radius")
# p.set_ylabel("Surface Density")
# sns.lineplot(x=rad, y=inter)
# plt.show()

#-----------------------Final expressions------------------------------#
G = c.G
Mplanck = np.sqrt(c.hbar*c.c/c.G)

vobs = example_df["Vobs"].values *u.km/u.s

def rho(R, Q):
    return rho_gas(R) + 0.5*Q*rho_disk_final(R) + 0.7*Q*rho_buldge_final(R)

def mass(R, z, Q):
    return 4/3 * np.pi * (R**2+z**2)**(3/2) * rho(R, Q)

def v(Rs, z, Q):
    alpha = 5.7
    lamb = 0.05
    if hasattr(Rs, '__len__'):
        return [np.sqrt(R*(G*mass(R, z, Q) + np.sqrt((alpha**3 * lamb**2/ Mplanck)* G * mass(R, z, Q)))) for R in Rs]
    else:
        return np.sqrt(Rs*(G*mass(Rs, z, Q) + np.sqrt((alpha**3 * lamb**2/ Mplanck)* G * mass(Rs, z, Q))))

sopt, scov = curve_fit(v, rad, vobs, bounds=(1e-2, 15)) #bounds from Hossenfelder on Q*


def v_interpolated(R):
    return v(R, sopt[0])

v_inter = [v_interpolated(R) for R in rad]

rc = sns.lineplot(x=rad, y=v_inter, label = "Model")
sns.scatterplot(x=rad, y=vobs, label = "Vobs data")
rc.set_xlabel("Radius")
rc.set_ylabel("Velocity")
rc.legend()
plt.show()
