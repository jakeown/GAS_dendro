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
mol = 'C2S'
#mol = 'HC5N'

# Convolve with 64 arcsec beam
convolve = False #arcseconds

#peak_channels
if mol=='C2S':
	peak_channels = [222,270] #C2S
else:
	peak_channels = [402,460] #HC5N
	

# Observed Region
region = 'Cepheus_L1251'

SN_thresh = 3.0

cube = SpectralCube.read(direct + region + '_' + mol + '_base_DR2.fits')
cube_km = cube.with_spectral_unit(u.km / u.s, velocity_convention='radio') 
subcube = cube_km.spectral_slab(-6 * u.km / u.s, -2 * u.km / u.s)
moment_0 = subcube.moment(order=0)
moment_1 = subcube.moment(order=1)
moment_2 = subcube.moment(order=2)
sigma_map = subcube.linewidth_sigma()
fwhm_map = subcube.linewidth_fwhm()
moment_0.write(direct + region + '_' + mol + '_moment0.fits', overwrite=True)
moment_1.write(direct + region + '_' + mol + '_moment1.fits', overwrite=True)
moment_2.write(direct + region + '_' + mol + '_moment2.fits', overwrite=True)
sigma_map.write(direct + region + '_' + mol + '_sigma.fits', overwrite=True)
fwhm_map.write(direct + region + '_' + mol + '_fwhm.fits', overwrite=True)

# Convolve map with larger beam?
if convolve!=False:
	beam = radio_beam.Beam(major=convolve*u.arcsec, minor=convolve*u.arcsec, pa=0*u.deg)
	cube2 = cube_km.convolve_to(beam)
	cube2.write(direct + region + '_' + mol + '_conv.fits', format='fits', overwrite=True)
else:
	cube2 = SpectralCube.read(direct + region + '_' + mol + '_conv.fits')
	subcube = cube2.spectral_slab(-6 * u.km / u.s, -2 * u.km / u.s)
	sub_cube = cube2[peak_channels[0]:peak_channels[1], :, :]
	moment_0 = subcube.moment(order=0)
	moment_1 = subcube.moment(order=1)
	moment_2 = subcube.moment(order=2)
	sigma_map = subcube.linewidth_sigma()
	fwhm_map = subcube.linewidth_fwhm()
	moment_0.write(direct + region + '_' + mol + '_moment0_conv.fits', overwrite=True)
	moment_1.write(direct + region + '_' + mol + '_moment1_conv.fits', overwrite=True)
	moment_2.write(direct + region + '_' + mol + '_moment2_conv.fits', overwrite=True)
	sigma_map.write(direct + region + '_' + mol + '_sigma_conv.fits', overwrite=True)
	fwhm_map.write(direct + region + '_' + mol + '_fwhm_conv.fits', overwrite=True)

header = sub_cube.header
cube_gauss = np.array(sub_cube.unmasked_data[:,:,:])
cube_gauss2 = np.array(sub_cube.unmasked_data[:,:,:])
shape = np.shape(sub_cube)

param_cube = cube_gauss[0]
param_cube = param_cube.reshape((1,) + param_cube.shape)
param_cube = np.concatenate((param_cube, param_cube, param_cube, param_cube, param_cube, param_cube), axis=0)
param_header = sub_cube.header

def p_eval2(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

spectra_x_axis_kms = np.array(sub_cube.spectral_axis) 

# Create some arrays full of NANs
# To be used in output cubes if fits fail
nan_array=np.empty(shape[0])
nan_array[:] = np.NAN
nan_array2=np.empty(param_cube.shape[0])
nan_array2[:] = np.NAN

# Loop through each pixel and find those
# with SNR above SN_thresh
x = []
y = []
pixels = 0
for (i,j), value in np.ndenumerate(cube_gauss[0]):
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

# Define a Gaussian fitting function for each pixel
# i, j are the x,y corrdinates of the pixel being fit
def pix_fit(i,j):
	spectra_full = np.array(cube2.unmasked_data[:,i,j])
	spectra = np.array(sub_cube.unmasked_data[:,i,j])
	Tpeak = max(spectra)
	vpeak = spectra_x_axis_kms[np.where(spectra==Tpeak)]
	rms = np.std(np.append(spectra_full[0:(peak_channels[0]-1)], spectra_full[(peak_channels[1]+1):len(spectra_full)]))
	err1 = np.zeros(shape[0])+rms
	noise=np.random.uniform(-1.*rms,rms,len(spectra_x_axis_kms))
	p2 = [Tpeak, vpeak, 0.3]
	try:
		coeffs, covar_mat = curve_fit(p_eval2, xdata=spectra_x_axis_kms, ydata=spectra, p0=p2, sigma=err1, maxfev=500)
		gauss = np.array(p_eval2(spectra_x_axis_kms,coeffs[0], coeffs[1], coeffs[2]))
		noisy_gauss = np.array(p_eval2(spectra_x_axis_kms,coeffs[0], coeffs[1], coeffs[2]))+noise
		params = np.append(coeffs, (covar_mat[0][0]**0.5, covar_mat[1][1]**0.5, covar_mat[2][2]**0.5))

		# Don't accept fit if fitted parameters are unreliable or too uncertain
		if (params[0] < 0.01) or (params[3] > 1.0) or (params[2] < 0.05) or (params[5] > 0.5) or (params[4] > 0.75):
			noisy_gauss = nan_array
			gauss = nan_array
			params = nan_array2

		# Don't accept fit if the SNR for fitted spectrum is less than SNR threshold
		if max(gauss)/rms < SN_thresh:
			noisy_gauss = nan_array
			gauss = nan_array
			params = nan_array2
		
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

cubey = SpectralCube(data=cube_gauss, wcs=sub_cube.wcs, header=sub_cube.header)
cubey.write(direct + 'gauss_cube_noise_' + mol + '_conv.fits', format='fits', overwrite=True)
cubey = SpectralCube(data=cube_gauss2, wcs=sub_cube.wcs, header=sub_cube.header)
cubey.write(direct + 'gauss_cube_' + mol + '_conv.fits', format='fits', overwrite=True)
cubey_moment0 = cubey.moment(order=0)
cubey_moment0.write(direct + region + '_' + mol + '_gauss_moment0.fits', overwrite=True)

param_header['NAXIS3'] = len(nan_array2)
param_header['WCSAXES'] = 3
param_header['CRPIX3'] = 1
param_header['CDELT3'] = 1
param_header['CRVAL3'] = 0
param_header['PLANE1'] = 'Tpeak'
param_header['PLANE2'] = 'VLSR'
param_header['PLANE3'] = 'sigma'
param_header['PLANE5'] = 'Tpeak_err'
param_header['PLANE6'] = 'VLSR_err'
param_header['PLANE7'] = 'sigma_err'

fits.writeto(direct + 'param_cube_' + mol + '_conv.fits', param_cube, header=param_header, clobber=True)



