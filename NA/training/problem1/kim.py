import numpy as np
from datetime import datetime
import pymsis

R_earth = 6378.0 
pi = 3.141592 
r_start = R_earth 
r_end = R_earth + 1000 
n = 10000

def density(alt, lat, lon): 

 date = np.datetime64(datetime(2025, 1, 1))
 
 lat = np.deg2rad(lat) 
 lon = np.deg2rad(lon)  

 data = pymsis.calculate(date, lon, lat, alt, version='2.1')
 rho = data[0, 0]   
 return float(rho)

r_i, r_f = r_start, r_end
theta_i, theta_f = 0, pi 
phi_i, phi_f = 0, 2*pi

r_s = np.linspace(r_i, r_f, 100)  
theta_s = np.linspace(theta_i, theta_f, 100)  
phi_s = np.linspace(phi_i, phi_f, 100)

delta_r = (r_f - r_i)/len(r_s)
delta_theta = (theta_f - theta_i)/len(theta_s)
delta_phi = (phi_f - phi_i)/len(phi_s)
I = 0

def f(r, theta):
 return rho*r**2*np.sin(theta)
