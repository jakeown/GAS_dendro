from spectral_cube import SpectralCube
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
from astropy.convolution import convolve
import radio_beam

# Directory for files
direct = '/Users/jkeown/Desktop/GAS_dendro/'

# Observed Molecule
#mol = 'C2S'
mol = 'HC5N'

#peak_channels
#peak_channels = [222,270] #C2S
peak_channels = [402,460] #HC5N

# Observed Region
region = 'Cepheus_L1251'

SN_thresh = 3.0

ckms=2.99792458*10**5

cube = SpectralCube.read(direct + region + '_' + mol + '_base_DR2.fits')

cube2 = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')  

# Convolve map with larger beam
beam = radio_beam.Beam(major=64*u.arcsec, minor=64*u.arcsec, pa=0*u.deg)
cube2 = cube2.convolve_to(beam)

#cube2.write('newfile2.fits', format='fits', overwrite=True)
#cube2 = SpectralCube.read('newfile2.fits')

# Need to load RMS map to use for SNR
cube = fits.getdata(direct + region + '_' + mol + '_base_DR2.fits')
header = cube2.header
shape = np.shape(cube)

cube_gauss = np.array(cube2.unmasked_data[:,:,:])
cube_gauss2 = np.array(cube2.unmasked_data[:,:,:])

rms_cube = fits.getdata(direct + region + '_' + mol + '_base_DR2_rms.fits')
#rms_mean = np.mean(rms_cube)
#rms_std = np.std(rms_cube)

rest_freq = header['RESTFRQ'] #transition frequency of given tracer (in 1/s)

param_cube = fits.getdata(direct + region + '_' + mol + '_base_DR2_rms.fits')
param_cube = param_cube.reshape((1,) + param_cube.shape)
param_cube = np.concatenate((param_cube, param_cube, param_cube, param_cube, param_cube, param_cube), axis=0)

param_header = fits.getheader(direct + region + '_' + mol + '_base_DR2.fits')

def p_eval2(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

p2 = [3., -4.0, 0.3]

#cube = SpectralCube.read('datacube2_ch1_ch2.fits')
#mask=(cube>1.3*u.K)& (cube<100.*u.K)

spectra_x_axis_kms = np.array(cube2.spectral_axis) 

nan_array=np.empty(shape[0])
nan_array[:] = np.NAN

nan_array2=np.empty(param_cube.shape[0])
nan_array2[:] = np.NAN

x = []
y = []
pixels = 0
for (i,j), value in np.ndenumerate(cube[0]):
     spectra=np.array(cube2.unmasked_data[:,i,j])
     rms = np.std(np.append(spectra[0:(peak_channels[0]-1)], spectra[(peak_channels[1]+1):len(spectra)]))
     if (max(spectra[peak_channels[0]:peak_channels[1]]) / rms) > SN_thresh:
            pixels+=1
	    x.append(i)
	    y.append(j)
     else:
	    cube_gauss[:,i,j]=nan_array
	    param_cube[:,i,j]=nan_array2
	    cube_gauss2[:,i,j]=nan_array
print str(pixels) + ' Pixels above SNR=' + str(SN_thresh) 

def pix_fit(i,j):
	spectra = np.array(cube2.unmasked_data[:,i,j])
	Tpeak = max(spectra[peak_channels[0]:peak_channels[1]])
	vpeak = spectra_x_axis_kms[peak_channels[0]:peak_channels[1]][np.where(spectra[peak_channels[0]:peak_channels[1]]==Tpeak)]
	rms = np.std(np.append(spectra[0:(peak_channels[0]-1)], spectra[(peak_channels[1]+1):len(spectra)]))
	err1 = np.zeros(shape[0])+rms
	noise=np.random.uniform(-1.*rms,rms,len(spectra_x_axis_kms))
	p2 = [Tpeak, vpeak, 0.3]
	p3 = [Tpeak, vpeak, 0.3, 1.0]
	try:
		#coeffs, covar_mat = curve_fit(p_eval3, xdata=spectra_x_axis_kms, ydata=spectra, p0=p3, sigma=err1, maxfev=500)
		#gauss = np.array(p_eval3(spectra_x_axis_kms,coeffs[0], coeffs[1], coeffs[2], coeffs[3]))
		#noisy_gauss = np.array(p_eval3(spectra_x_axis_kms,coeffs[0], coeffs[1], coeffs[2], coeffs[3]))+noise
		#params = np.append(coeffs, (covar_mat[0][0]**0.5, covar_mat[1][1]**0.5, covar_mat[2][2]**0.5, covar_mat[3][3]**0.5))

		coeffs, covar_mat = curve_fit(p_eval2, xdata=spectra_x_axis_kms, ydata=spectra, p0=p2, sigma=err1, maxfev=500)
		gauss = np.array(p_eval2(spectra_x_axis_kms,coeffs[0], coeffs[1], coeffs[2]))
		noisy_gauss = np.array(p_eval2(spectra_x_axis_kms,coeffs[0], coeffs[1], coeffs[2]))+noise
		params = np.append(coeffs, (covar_mat[0][0]**0.5, covar_mat[1][1]**0.5, covar_mat[2][2]**0.5))

		#if (coeffs[0] < 0.) or (coeffs[3] < 0.):
		#	noisy_gauss = nan_array
		#	gauss = nan_array
		#	params = nan_array2
		
	except RuntimeError:
		noisy_gauss = nan_array
		gauss = nan_array
		params = nan_array2
	
	#plt.plot(spectra_x_axis_kms, spectra, color='blue', drawstyle='steps')
	#plt.plot(spectra_x_axis_kms, gauss, color='red')
	#plt.show()
	#plt.close()
	return i, j, noisy_gauss, params, gauss

# Parallel computation:
nproc = 3  	# maximum number of simultaneous processes desired
queue = pprocess.Queue(limit=nproc)
calc = queue.manage(pprocess.MakeParallel(pix_fit))
#results = pprocess.Map(limit=nproc, reuse=1)
#parallel_function = results.manage(pprocess.MakeReusable(pix_fit))
tic=time.time()
counter = 0

#for i,j in zip(x,y):
#	pix_fit(i,j)

for i,j in zip(x,y):
	calc(i,j)
for i,j,result,parameters,r_gauss in queue:
	cube_gauss[:,i,j]=result
	param_cube[:,i,j]=parameters
	cube_gauss2[:,i,j]=r_gauss
	counter+=1
	print str(counter) + ' of ' + str(pixels) + ' pixels completed'
print "%f s for parallel computation." % (time.time() - tic)

cubey = SpectralCube(data=cube_gauss, wcs=cube2.wcs, header=cube2.header)
cubey.write(direct + 'gauss_cube_noise_' + mol + 'conv.fits', format='fits', overwrite=True)
cubey = SpectralCube(data=cube_gauss2, wcs=cube2.wcs, header=cube2.header)
cubey.write(direct + 'gauss_cube_' + mol + 'conv.fits', format='fits', overwrite=True)

param_header['NAXIS3'] = len(nan_array2)
param_header['WCSAXES'] = 3
param_header['CRPIX3'] = 1
param_header['CDELT3'] = 1
param_header['CRVAL3'] = 0
param_header['PLANE1'] = 'TaTau'
param_header['PLANE2'] = 'VLSR'
param_header['PLANE3'] = 'FWHM'
param_header['PLANE4'] = 'tau_main'
param_header['PLANE5'] = 'TaTau_err'
param_header['PLANE6'] = 'VLSR_err'
param_header['PLANE7'] = 'FWHM_err'
param_header['PLANE8'] = 'tau_main_err'

fits.writeto(direct + 'param_cube_' + mol + 'conv.fits', param_cube, header=param_header, clobber=True)

#counter=0
#tic=time.time()
#for i,j in zip(x,y):
#	pix_fit(i,j)
#	counter+=1
#	print str(counter) + ' of ' + str(pixels) + ' completed'
#print "%f s for linear computation." % (time.time() - tic)

