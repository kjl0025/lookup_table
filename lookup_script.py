import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

try:
    from emccd_detect.emccd_detect import EMCCDDetect
except:
    pass #need this module to add detector noise

'''This script goes through the analysis the paper goes through: generates Gaussians 
of various non-integral centroid positions to make a lookup table, generates randomly positioned 
Gaussians for testing, centroids them using the lookup table method and the fiting algorithm methods, 
and compares the true simulated positions with the results of the two methods.  
--If one_to_one is set to True, the one-to-one analysis is carried out 
instead, for which Gaussians are made at unique positions shifted slightly away from the lookup table positions 
and fitted using the lookup table to see how many unique centroids are found.  
--If paper_plot is True, the plot in the paper is plotted.  
--If the parameter random_amp is True, the amplitude of the test PSFs is randomized.
--If the parameter noise is True, detector noise is simulated for the test PSFs. If one_to_one is also True, noise will be forced to be False in the script.
--The parameter step controls the step size of the lookup table.
--The parameter sigma controls the standard deviation (same for x and y) of the Gaussians.
--The parameter save should be True if the user wants to save the lookup table (especially if it takes a while to make).
--The parameter path is the path for saving the lookup table (if save is True).  If '', it is saved in the current directory.
--The parameter load should be True if the user wants to load in a dictionary in the directory specified by path (assuming the filename scheme used when save is True).
--The parameter full_status should be True if the user wants the fitting algorithm to fit all six Gaussian parameters (C, A, x0, y0, sx, sy).  If False, only A, x0, and y0 are fitted, 
in keeping with the 3 parameters handled by the lookup table method, and if the 3-parameter fit fails for a given PSF, a 2-parameter fit (for x0 and y0) is attempted.

Other parameters, such as sigma of the Gaussians and the step size 
for the lookup table, can be changed after after if __name__ == '__main__'.

By default, this makes 15x15 frames containing a single PSF and uses centroids made 
with centroids between 7.5 and 8.5.'''

one_to_one = False
paper_plot = False
random_amp = True
noise = True # detector noise simulated using emccd_detect
step = 0.1 # pixels
sigma = 0.4 # pixels
path = ''
save = False
load = False
full_status = True

def gauss_fit_full(data, num_bad_fits):
    '''Fits a 2D array (data) to a Gaussian and returns the best-fit parameters and the 
     array containing the output of the Gaussian function with the 6 best-fit parameters.'''
    Y = np.arange(0,len(data))
    X = np.arange(0,len(data[0]))
    X, Y = np.meshgrid(X,Y)
    XY = np.vstack((X.ravel(), Y.ravel()))
    init_guess = (200, 5000, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, 0, 0, 0, 0]
    bounds = (lb, ub)
    try:
        popt, pcov = curve_fit(gauss_spot_full, XY, data.ravel(), bounds=bounds, p0=init_guess) #, maxfev=1e7, xtol=1e-15)
    except:
        popt = init_guess
        num_bad_fits += 1
    gauss_val = gauss_spot_full(XY, *popt)
    return *popt, gauss_val, num_bad_fits

def gauss_fit(data, num_bad_fits):
    '''Fits a 2D array (data) to a Gaussian and returns the best-fit parameters and the 
     array containing the output of the Gaussian function with the 3 best-fit parameters.'''
    Y = np.arange(0,len(data))
    X = np.arange(0,len(data[0]))
    X, Y = np.meshgrid(X,Y)
    XY = np.vstack((X.ravel(), Y.ravel()))
    init_guess = (5000, len(data[0])/2, len(data)/2)
    ub = [2**16-1, len(data[0]), len(data)]
    lb = [0, 0, 0]
    bounds = (lb, ub)
    try:
        popt, pcov = curve_fit(gauss_spot, XY, data.ravel(), bounds=bounds, p0=init_guess) #, maxfev=1e7, xtol=1e-15)
    except:
        popt = init_guess
        num_bad_fits += 1
    gauss_val = gauss_spot(XY, *popt)
    return 0, popt[0], popt[1], popt[2], sigma, sigma, gauss_val, num_bad_fits

def gauss_fit_2(data, num_bad_fits):
    '''Fits a 2D array (data) to a Gaussian and returns the best-fit parameters and the 
     array containing the output of the Gaussian function with the 2 best-fit parameters.'''
    Y = np.arange(0,len(data))
    X = np.arange(0,len(data[0]))
    X, Y = np.meshgrid(X,Y)
    XY = np.vstack((X.ravel(), Y.ravel()))
    init_guess = (len(data[0])/2, len(data)/2)
    ub = [len(data[0]), len(data)]
    lb = [0, 0]
    bounds = (lb, ub)
    try:
        popt, pcov = curve_fit(gauss_spot_2, XY, data.ravel(), bounds=bounds, p0=init_guess)#, maxfev=1e7, xtol=1e-15)
    except:
        popt = init_guess
        num_bad_fits += 1
    gauss_val = gauss_spot_2(XY, *popt)
    return 0, 0.15*ceil, popt[0], popt[1], sigma, sigma, gauss_val, num_bad_fits

def fit_algorithm(full_frame, pkRow, pkCol, sub_frame_size, full_status, num_bad_fits):
    '''For an input 2D array (full_frame) containing one or more PSFs, this function 
    finds the Gaussian centroid of a PSF with brightest pixel located at the row index pkRow and 
    column index pkCol.  It performs the fit for that PSF within a sub-frame square region of side
    length sub_frame_size, in pixels.'''
    centroid = {}
    nr = len(full_frame)
    nc = len(full_frame[0])
    rowMin = int(max(0, pkRow - np.floor(sub_frame_size/2)))
    rowMax = int(min(nr-1, pkRow + np.floor(sub_frame_size/2)))
    colMin = int(max(0, pkCol - np.floor(sub_frame_size/2)))
    colMax = int(min(nc-1, pkCol + np.floor(sub_frame_size/2)))

    image = full_frame[rowMin:rowMax+1, colMin:colMax+1]
    
    if full_status:
        offset, A2, xc2, yc2, sigmax2, sigmay2, SigEst2, num_bad_fits_after = gauss_fit_full(image, num_bad_fits)
    else:
        offset, A2, xc2, yc2, sigmax2, sigmay2, SigEst2, num_bad_fits_after = gauss_fit(image, num_bad_fits)
        if num_bad_fits_after > num_bad_fits:
            offset, A2, xc2, yc2, sigmax2, sigmay2, SigEst2, num_bad_fits_after = gauss_fit_2(image, num_bad_fits)
    if np.isnan(float(xc2)) or np.isnan(float(yc2)) or float(xc2)<0 or float(yc2)<0 or float(xc2)>image.shape[1]-1 or float(yc2)>image.shape[0]-1 or \
    np.isnan(float(sigmax2)) or np.isnan(float(sigmay2)) or np.isnan(float(A2)):
        print("Even a true Gaussian fit gives NaNs")
    centroid['row'] = yc2 + rowMin
    centroid['col'] = xc2 + colMin
    centroid['GauImg'] = SigEst2
    centroid['GauAmp'] = A2
    centroid['GauSig'] = np.sqrt(sigmax2**2 + sigmay2**2)
    centroid['GauSigx'] = sigmax2
    centroid['GauSigy'] = sigmay2
    centroid['offset'] = offset

    return centroid, num_bad_fits_after

def gauss_spot_full(xy, offset, A, x0, y0, sx, sy):
    '''2D Gaussian function.  
    Parameters:
        xy: vector stack of x and y points 
        offset: pedestal level the Gaussian sits above
        A: amplitude
        x0: x coordinate of center of Gaussian
        y0: y coordinate of center of Gaussian
        sx: standard deviation in x direction
        sy: standard deviation in y direction
    '''
    (x, y) = xy
    return offset + A*np.e**(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2)))

def gauss_spot(xy, A, x0, y0):
    '''2D Gaussian function.  
    Parameters:
        xy: vector stack of x and y points 
        offset: pedestal level the Gaussian sits above
        A: amplitude
        x0: x coordinate of center of Gaussian
        y0: y coordinate of center of Gaussian
        sx: standard deviation in x direction
        sy: standard deviation in y direction
    '''
    offset = 0
    sx = sigma
    sy = sigma
    (x, y) = xy
    return offset + A*np.e**(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2)))

def gauss_spot_2(xy, x0, y0):
    '''2D Gaussian function.  
    Parameters:
        xy: vector stack of x and y points 
        offset: pedestal level the Gaussian sits above
        A: amplitude
        x0: x coordinate of center of Gaussian
        y0: y coordinate of center of Gaussian
        sx: standard deviation in x direction
        sy: standard deviation in y direction
    '''
    offset = 0
    sx = sigma
    sy = sigma
    A = 0.15*ceil
    (x, y) = xy
    return offset + A*np.e**(-((x-x0)**2/(2*sx**2) + (y-y0)**2/(2*sy**2)))

def int_gauss(offset, A, x0, y0, sx, sy, x1, x2, y1, y2):
    '''Double integral of Gaussian function with bounds x1, x2 and y1, y2. The physical pixel center is
    specified by x0, y0. This function is used for 
    finding the value for each pixel of a simulated Gaussian.'''
    pix_val = (offset*(x1 - x2)*(y1 - y2) + 
        (A*np.pi*sx*sy*(erf((-x0 + x1)/(np.sqrt(2)*sx)) - erf((-x0 + x2)/(np.sqrt(2)*sx)))*
        (erf((-y0 + y1)/(np.sqrt(2)*sy)) - erf((-y0 + y2)/(np.sqrt(2)*sy))))/2.)
    return pix_val
    

if __name__ == '__main__':

    M = 15 #window of 15x15 surrounding the PSF
    np.random.seed(456) # to make reproducible noise simulations
    if one_to_one:
        noise = False
    e_per_dn = 1/.13
    ceil = (2**12 - 6)*e_per_dn #in electrons

    lookup_distance_residuals_list = []
    centroid_distance_residuals_list = []
    x_diff_list = []
    y_diff_list = []
    x_fitting_diff_list = []
    y_fitting_diff_list = []
    num_bad_fits = 0

    if paper_plot:
        plots = []
        up = - np.inf
        low = np.inf
        sigmas = [0.2, 0.4, 0.8, 2]
        for sigma in sigmas:
            x = np.arange(M)
            y = np.arange(M)
            X,Y = np.meshgrid(x,y)
            scale = 0.8
            g2 = np.zeros((M,M))
            for j in x:
                for k in y:
                    pix_val = int_gauss(0, ceil*scale, 7.5, 7.5, sigma, sigma, j-0.5, j+0.5, k-0.5, k+0.5)
                    g2[k,j] += pix_val
            binned_g2 = g2
            binned_g2n_arr = np.zeros([15, 15, 5])
            dark_arr = np.zeros([15, 15, 5])
            for k in range(5): 
                try:
                    emccd = EMCCDDetect(em_gain=1, full_well_image=ceil, full_well_serial=1e12, dark_current=50, cic=0, read_noise=9, bias=2000, qe=0.7, cr_rate=0, pixel_pitch=10e-6, eperdn=e_per_dn, nbits=12, numel_gain_register=604, meta_path=None)
                except:
                    raise Exception('emccd_detect module needed to add noise.')
                binned_g2n_arr[:, :, k] = emccd.sim_sub_frame(binned_g2, 0.05).astype('int64')
                dark_arr[:, :, k] = emccd.sim_sub_frame(np.zeros_like(binned_g2n_arr[:, :, k]), 0.05).astype('int64')
            binned_g2n = np.mean(binned_g2n_arr, axis=2)
            dark = np.mean(dark_arr, axis=2)
            binned_g2 = np.subtract(binned_g2n, dark)
            plots.append(binned_g2)
            if binned_g2.max() > up:
                up = binned_g2.max()
            if binned_g2.min() < low:
                low = binned_g2.min()  
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(plots[0], vmin=low, vmax=up)
        ax2.imshow(plots[1], vmin=low, vmax=up)
        ax3.imshow(plots[2], vmin=low, vmax=up)
        im = ax4.imshow(plots[3], vmin=low, vmax=up)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

    if one_to_one: # makes 100 PSFs for testing injectivity
        x_i_array = np.arange(7.5, 8.5, step) + 0.3*step
        y_i_array = np.arange(7.5, 8.5, step) + 0.3*step
    else: # 900 PSFs made here for testing the lookup table method
        x_i_array = np.random.uniform(7.5, 8.5, 30)
        y_i_array = np.random.uniform(7.5, 8.5, 30) 

    ### Create the lookup table
    if not load:
        lookup_dict = {}
        
        for x_i in np.arange(7.5, 8.5025, step):
            for y_i in np.arange(7.5, 8.5025, step):
                x_pts = np.array([x_i]) # roughly in the middle of a 15x15 frame
                y_pts = np.array([y_i])

                binned_g2 = np.zeros((M,M))

                for i in range(len(x_pts)):
                    x = np.arange(M)
                    y = np.arange(M)
                    X,Y = np.meshgrid(x,y)
                    for j in x:
                        for k in y:
                            pix_val = int_gauss(0, ceil*.8, x_pts[i], y_pts[i], sigma, sigma, j-0.5, j+0.5, k-0.5, k+0.5)
                            binned_g2[k,j] += pix_val
        
                true_x = x_pts 
                true_y = y_pts 
                
                binned_g2_tuple = tuple(map(tuple, binned_g2))
                lookup_dict[binned_g2_tuple] = [float(true_x), float(true_y)]
    else:
        saved_dict = np.load(path+'lookup_table_sigma'+str(sigma)+'_step'+str(step)+'.npy', allow_pickle=True)
        lookup_dict = dict(saved_dict[()])
    if save:
        np.save(path+'lookup_table_sigma'+str(sigma)+'_step'+str(step)+'.npy', lookup_dict, allow_pickle=True, fix_imports=True)
    
    ### Now to test the lookup table

    est_pos_list = []
        
    for x_i in x_i_array:
        for y_i in y_i_array:
            x_pts = np.array([x_i]) # roughly in the middle of a 15x15 frame
            y_pts = np.array([y_i])

            binned_g2 = np.zeros((M,M))

            for i in range(len(x_pts)):
                x = np.arange(M)
                y = np.arange(M)
                X,Y = np.meshgrid(x,y)

                # randomized amplitude
                if random_amp:
                    scale = np.random.uniform(0.3,0.9)
                else:
                    scale = 0.8
                for j in x:
                    for k in y:
                        pix_val = int_gauss(0, ceil*scale, x_pts[i], y_pts[i], sigma, sigma, j-0.5, j+0.5, k-0.5, k+0.5)
                        binned_g2[k,j] += pix_val

            signal_map = binned_g2.copy()
            binned_g2n = np.zeros_like(signal_map) # stays 0 if no noise added
            SNR_list = []
            three_sig_SNR_list = []
            if noise:
                binned_g2n_arr = np.zeros([15, 15, 5])
                dark_arr = np.zeros([15, 15, 5])
                for k in range(5): 
                    emccd = EMCCDDetect(em_gain=1, full_well_image=ceil, full_well_serial=1e12, dark_current=50, cic=0, read_noise=9, bias=2000, qe=0.7, cr_rate=0, pixel_pitch=10e-6, eperdn=e_per_dn, nbits=12, numel_gain_register=604, meta_path=None)
                    binned_g2n_arr[:, :, k] = emccd.sim_sub_frame(binned_g2, 0.05).astype('int64')
                    dark_arr[:, :, k] = emccd.sim_sub_frame(np.zeros_like(binned_g2n_arr[:, :, k]), 0.05).astype('int64')
                binned_g2n = np.mean(binned_g2n_arr, axis=2)
                dark = np.mean(dark_arr, axis=2)
                binned_g2 = np.subtract(binned_g2n, dark)
                #multiply photon map by exposure time and quantum efficiency to get electrons, then divide by e-/DN to get DN units
                DN_signal_map = signal_map*0.05*0.7/e_per_dn
                noise_map = binned_g2n - DN_signal_map 
                SNR_max = np.max(signal_map/noise_map)
                st_row = int(max(0, np.floor(y_pts-3*sigma)))
                end_row = int(min(15, np.ceil(y_pts+3*sigma)))
                st_col = int(max(0, np.floor(x_pts-3*sigma)))
                end_col = int(min(15, np.ceil(x_pts+3*sigma)))
                three_sig_SNR = np.sum(signal_map[st_row:end_row, st_col:end_col])/np.sum(noise_map[st_row:end_row, st_col:end_col])
                SNR = np.sum(signal_map)/np.sum(noise_map)
                SNR_list.append(SNR)
                three_sig_SNR_list.append(three_sig_SNR)

            pk = np.unravel_index(np.argmax(binned_g2), binned_g2.shape)
            centroid, num_bad_fits = fit_algorithm(binned_g2, pk[0], pk[1], binned_g2.shape[0]+1, full_status, num_bad_fits)


            true_x = x_pts 
            true_y = y_pts 

            true_pos = np.reshape(np.asarray([true_x, true_y]), 2)
            centroid_pos = np.reshape(np.asarray([centroid['col'], centroid['row']]), 2)
            
            residuals_dict = {}

            for key in lookup_dict.keys():
                m = tuple(map(tuple, key))
                binned_g2 = (binned_g2*np.sum(key))/np.sum(binned_g2)
                residual = float(np.sum(np.square(np.subtract(binned_g2, key))))
                residuals_dict[residual] = m

            min_residual = min(residuals_dict.keys())
            est_pos = np.asarray(lookup_dict[residuals_dict[min_residual]])
            est_pos_list.append(est_pos)
            x_diff_list.append(true_pos[0]-est_pos[0])
            y_diff_list.append(true_pos[1]-est_pos[1])
            x_fitting_diff_list.append(true_pos[0]-centroid_pos[0])
            y_fitting_diff_list.append(true_pos[1]-centroid_pos[1])
            lookup_distance_residuals_list.append(np.square(np.subtract(est_pos[0], true_pos[0])+np.square(np.subtract(est_pos[1], true_pos[1]))))
            centroid_distance_residuals_list.append(np.square(np.subtract(centroid_pos[0], true_pos[0]))+np.square(np.subtract(centroid_pos[1], true_pos[1])))
    
    print("Step size used: ", step)
    print('Sigma simulated: ', sigma)
    
    if paper_plot:
        plt.hist(x_diff_list,30); plt.title('Histogram of Lookup Table Error in x');plt.xlabel('True x - Lookup Table x (pixels)'); plt.ylabel('Frequency')
        plt.figure(); plt.hist(y_diff_list,30);plt.title('Histogram of Lookup Table Error in y');plt.xlabel('True y - Lookup Table y (pixels)'); plt.ylabel('Frequency')
        plt.figure(); plt.hist(x_fitting_diff_list,30); plt.title('Histogram of Fitting Algorithm Error in x');plt.xlabel('True x - Fitting Algorithm x (pixels)'); plt.ylabel('Frequency')
        plt.figure(); plt.hist(y_fitting_diff_list,30);plt.title('Histogram of Fitting Algorithm Error in y');plt.xlabel('True y - Fitting Algorithm y (pixels)'); plt.ylabel('Frequency')
        plt.show()
    if one_to_one:
        print('Number of estimated positions chosen: ', len(est_pos_list))
        print('Number of unique estimated positions chosen: ', np.unique(np.array(est_pos_list),axis=0).shape[0])
        dist_from_true = np.sqrt(lookup_distance_residuals_list)
        print('Number of times a test PSF was matched to a lookup table centroid that is not closest to that of the test PSF: ', np.sum(dist_from_true > np.sqrt(2)*0.3*step))
    else:
        est_RMSE = np.sqrt(np.mean(lookup_distance_residuals_list)) 

        cent_RMSE = np.sqrt(np.mean(centroid_distance_residuals_list))
        
        print('Lookup Table RMSE mean is {}'.format(est_RMSE))
        std_est_RMSE = np.sqrt(1/(2*len(lookup_distance_residuals_list)))*est_RMSE #np.std(est_RMSE_list)
        print('Lookup Table RMSE STD is {}'.format(std_est_RMSE))

        print('Fitting Algorithm RMSE mean is {}'.format(cent_RMSE))
        std_cent_RMSE = np.sqrt(1/(2*len(centroid_distance_residuals_list)))*cent_RMSE #np.std(cent_RMSE_list)
        print('Fitting Algorithm RMSE STD is {}'.format(std_cent_RMSE))

        if noise:
            print("Average SNR of PSFs over 15x15 area: ", np.mean(SNR_list))
            print("Average SNR of PSFs over 3-sigma area: ", np.mean(three_sig_SNR_list))
        print('Number of times traditional fitting algorithm could not find best fit: ', num_bad_fits)
    pass
