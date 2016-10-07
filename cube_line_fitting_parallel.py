#from spectral_cube import SpectralCube
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from scipy.optimize import curve_fit
import math
import pylab
from scipy import *
import time
import pprocess

SN_thresh = 3.0

ckms=2.99792458*10**5

voff_lines = np.array([19.8513, 
                  19.3159, 
                  7.88669, 
                  7.46967, 
                  7.35132, 
                  0.460409, 
                  0.322042, 
                  -0.0751680, 
                  -0.213003,  
                  0.311034, 
                  0.192266, 
                  -0.132382, 
                  -0.250923, 
                  -7.23349, 
                  -7.37280, 
                  -7.81526, 
                  -19.4117, 
                  -19.5500])

tau_wts = np.array([0.0740740, 
              0.148148, 
              0.0925930, 
              0.166667, 
              0.0185190, 
              0.0370370, 
              0.0185190, 
              0.0185190, 
              0.0925930, 
              0.0333330, 
              0.300000, 
              0.466667, 
              0.0333330, 
              0.0925930, 
              0.0185190, 
              0.166667, 
              0.0740740, 
              0.148148])

#deltanu = -1*voff_lines/((c/1000)*23.6944955e9)

def p_eval2(x, TaTau, Vlsr, FWHM, tau_main):

	#Define frequencies and relative intensities of 7 components
	#nu_lines=numpy.array(nu_lines_in)*10.**-3
	#rel_wts1 = numpy.array(rel_wts1_in)
	#rel_wts2 = rel_wts1/sum(rel_wts1)
	rel_wts = tau_wts
	#ckms = 2.99792458*(10**5)  #km s**-1
	#voff_lines1 = -1.0*ckms*(nu_lines - nu_lines[17])/nu_lines[4]
	#voff_lines = voff_lines1.tolist()
	N_comp = len(voff_lines)
	N_vel  = len(x)

	kb   = 1.380658E-16  # erg K**-1
	h    = 6.6260755E-27 # erg s
	Tb   = 2.73          # K
	T0   = h*154.2E9/kb   # GHz

	#Define the line for each velocity component
	tau_vi = np.zeros(shape=(N_comp, N_vel))
	tau_vi = np.zeros(shape=(N_comp, N_vel))
	for i in xrange(0, N_comp):
		for v in xrange(0, N_vel):
			tau_vi[i,v] = rel_wts[i]*np.exp(-4.0*math.log(2.0)*((x[v]-voff_lines[i]-Vlsr)/FWHM)**2.0)

	#Define the total tau function
	t_v = np.zeros(shape=N_vel)
	for v in xrange(0, N_vel):
		for i in xrange(0, N_comp):
			t_v[v] = t_v[v] + tau_vi[i,v]
	t_v = tau_main*t_v

	return TaTau/tau_main*(1.0 - exp(-1.0*t_v))

def p_eval3(x, TaTau, Vlsr, FWHM, tau_main):

	#Constants
	T_0 = 2.73 		#cosmic background temperature
	v1 = 23694495500.0 	#transition frequency of given tracer (in 1/s)
	h = (6.626*(10**-27))	#Planck's constant (in J*s)
	k = (1.381*(10**-16)) 	#Boltzmann constant (in J/Kelvin)
	T_1 = (h*v1)/k		#variable for planck corrected brightness temp. equations

	#Define tau component
	t_v = tau_main*np.exp(-4.0*math.log(2.0)*((x-Vlsr)/FWHM)**2.0)

	return TaTau/tau_main*(1.0 - exp(-1.0*t_v))

p3 = [3., -4.0, 0.3, 1.0]

#cube = SpectralCube.read('datacube2_ch1_ch2.fits')
#mask=(cube>1.3*u.K)& (cube<100.*u.K)

# Need to load RMS map to use for SNR
cube = fits.getdata('Cepheus_L1251_NH3_11_base_all.fits')
header = fits.getheader('Cepheus_L1251_NH3_11_base_all.fits')
shape = np.shape(cube)

cube_gauss = fits.getdata('Cepheus_L1251_NH3_11_base_all.fits')

rms_cube = fits.getdata('Cepheus_L1251_NH3_11_base_all_rms.fits')
#rms_mean = np.mean(rms_cube)
#rms_std = np.std(rms_cube)

freq_i = header['CRVAL3']
freq_step = header['CDELT3']
freq_ref = header['CRPIX3']

spectra_x_axis_start = freq_i-(freq_step*freq_ref)
spectra_x_axis_end = freq_i+(freq_step*(shape[0]-freq_ref))
spectra_x_axis = np.linspace(spectra_x_axis_start, spectra_x_axis_end, num=shape[0])
spectra_x_axis_kms = ckms*(1.-(spectra_x_axis/23694495500.0))

nan_array=np.empty(shape[0])
nan_array[:] = np.NAN

x = []
y = []
pixels = 0
for (i,j), value in np.ndenumerate(cube[0]):
     spectra=cube[:,i,j]
     rms = rms_cube[i,j]
     if (max(spectra) / rms) > SN_thresh:
            pixels+=1
	    x.append(i)
	    y.append(j)
     else:
	    cube_gauss[:,i,j]=nan_array
print str(pixels) + ' Pixels above SNR=' + str(SN_thresh) 

def pix_fit(i,j):
	spectra = cube[:,i,j]
	rms = rms_cube[i,j]
	err1 = np.zeros(shape[0])+rms
	noise=np.random.uniform(-1.*rms,rms,len(spectra_x_axis_kms))
	try:
		coeffs, covar_mat = curve_fit(p_eval2, xdata=spectra_x_axis_kms, ydata=spectra, p0=p3, sigma=err1, maxfev=250)
		noisy_gauss = np.array(p_eval3(spectra_x_axis_kms,coeffs[0], coeffs[1], coeffs[2], coeffs[3]))+noise
	except RuntimeError:
		noisy_gauss = nan_array
	return i,j,noisy_gauss

# Parallel computation:
nproc = 14  	# maximum number of simultaneous processes desired
queue = pprocess.Queue(limit=nproc)
calc = queue.manage(pprocess.MakeParallel(pix_fit))
#results = pprocess.Map(limit=nproc, reuse=1)
#parallel_function = results.manage(pprocess.MakeReusable(pix_fit))
tic=time.time()
counter = 0
for i,j in zip(x,y):
	calc(i,j)
for i,j,result in queue:
	cube_gauss[:,i,j]=result
	counter+=1
	print str(counter) + ' of ' + str(pixels) + ' pixels completed'
print "%f s for parallel computation." % (time.time() - tic)
fits.writeto('gauss_cube.fits', cube_gauss,clobber=True)

#counter=0
#tic=time.time()
#for i,j in zip(x,y):
#	pix_fit(i,j)
#	counter+=1
#	print str(counter) + ' of ' + str(pixels) + ' completed'
#print "%f s for linear computation." % (time.time() - tic)

