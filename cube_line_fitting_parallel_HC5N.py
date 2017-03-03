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

# Directory for files
direct = '/Users/jkeown/Desktop/GAS_dendro/'

# Observed Molecule
mol = 'HC5N'
# Observed Region
region = 'Cepheus_L1251'

SN_thresh = 3.0

ckms=2.99792458*10**5

# Need to load RMS map to use for SNR
cube = fits.getdata(direct + region + '_' + mol + '_base_DR2.fits')
header = fits.getheader(direct + region + '_' + mol + '_base_DR2.fits')
shape = np.shape(cube)

cube_gauss = fits.getdata(direct + region + '_' + mol + '_base_DR2.fits')

rms_cube = fits.getdata(direct + region + '_' + mol + '_base_DR2_rms.fits')
#rms_mean = np.mean(rms_cube)
#rms_std = np.std(rms_cube)

rest_freq = header['RESTFRQ'] #transition frequency of given tracer (in 1/s)

param_cube = fits.getdata(direct + region + '_' + mol + '_base_DR2_rms.fits')
param_cube = param_cube.reshape((1,) + param_cube.shape)
param_cube = np.concatenate((param_cube, param_cube, param_cube, param_cube, param_cube, param_cube, param_cube, param_cube), axis=0)

def p_eval3(x, TaTau, Vlsr, FWHM, tau_main):

	#Constants
	T_0 = 2.73 		#cosmic background temperature
	v1 = rest_freq  	#transition frequency of given tracer (in 1/s)
	h = (6.626*(10**-27))	#Planck's constant (in J*s)
	k = (1.381*(10**-16)) 	#Boltzmann constant (in J/Kelvin)
	T_1 = (h*v1)/k		#variable for planck corrected brightness temp. equations

	#Define tau component
	t_v = tau_main*np.exp(-4.0*math.log(2.0)*((x-Vlsr)/FWHM)**2.0)

	return TaTau/tau_main*(1.0 - exp(-1.0*t_v))

p3 = [3., -4.0, 0.3, 1.0]

#cube = SpectralCube.read('datacube2_ch1_ch2.fits')
#mask=(cube>1.3*u.K)& (cube<100.*u.K)

freq_i = header['CRVAL3']
freq_step = header['CDELT3']
freq_ref = header['CRPIX3']

spectra_x_axis_start = freq_i-(freq_step*freq_ref)
spectra_x_axis_end = freq_i+(freq_step*(shape[0]-freq_ref))
spectra_x_axis = np.linspace(spectra_x_axis_start, spectra_x_axis_end, num=shape[0])
spectra_x_axis_kms = ckms*(1.-(spectra_x_axis/rest_freq))

nan_array=np.empty(shape[0])
nan_array[:] = np.NAN

nan_array2=np.empty(param_cube.shape[0])
nan_array2[:] = np.NAN

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
	    param_cube[:,i,j]=nan_array2
print str(pixels) + ' Pixels above SNR=' + str(SN_thresh) 

def pix_fit(i,j):
	spectra = cube[:,i,j]
	rms = rms_cube[i,j]
	err1 = np.zeros(shape[0])+rms
	noise=np.random.uniform(-1.*rms,rms,len(spectra_x_axis_kms))
	try:
		coeffs, covar_mat = curve_fit(p_eval3, xdata=spectra_x_axis_kms, ydata=spectra, p0=p3, sigma=err1, maxfev=250)
		noisy_gauss = np.array(p_eval3(spectra_x_axis_kms,coeffs[0], coeffs[1], coeffs[2], coeffs[3]))+noise
		params = np.append(coeffs, (covar_mat[0][0]**0.5, covar_mat[1][1]**0.5, covar_mat[2][2]**0.5, covar_mat[3][3]**0.5))
	except RuntimeError:
		noisy_gauss = nan_array
		params = nan_array2
	return i, j, noisy_gauss, params

# Parallel computation:
nproc = 3  	# maximum number of simultaneous processes desired
queue = pprocess.Queue(limit=nproc)
calc = queue.manage(pprocess.MakeParallel(pix_fit))
#results = pprocess.Map(limit=nproc, reuse=1)
#parallel_function = results.manage(pprocess.MakeReusable(pix_fit))
tic=time.time()
counter = 0
for i,j in zip(x,y):
	calc(i,j)
for i,j,result,parameters in queue:
	cube_gauss[:,i,j]=result
	param_cube[:,i,j]=parameters
	counter+=1
	print str(counter) + ' of ' + str(pixels) + ' pixels completed'
print "%f s for parallel computation." % (time.time() - tic)

fits.writeto(direct + 'gauss_cube_' + mol + '.fits', cube_gauss,clobber=True)
fits.writeto(direct + 'param_cube_' + mol + '.fits', param_cube,clobber=True)

#counter=0
#tic=time.time()
#for i,j in zip(x,y):
#	pix_fit(i,j)
#	counter+=1
#	print str(counter) + ' of ' + str(pixels) + ' completed'
#print "%f s for linear computation." % (time.time() - tic)

