import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from emccd_detect.emccd_detect import EMCCDDetect

'''This script goes through the analysis the paper goes through: generates Gaussians 
of various non-integral centroid positions to make a lookup table, generates randomly positioned 
Gaussians for testing, centroids them using the lookup table method and the fiting algorithm methods, 
and compares the true simulated positions with the results of the two methods.  
--If one_to_one is set to True, the one-to-one analysis is carried out 
instead, for which Gaussians are made at unique positions shifted slightly away from the lookup table positions 
and fitted using the lookup table to see how many unique centroids are found.  
--If paper_plot is True, the plot in the paper is plotted.  
--The parameter step controls the step size of the lookup table.
--The parameter sigma controls the standard deviation (same for x and y) of the Gaussians.
--The parameter save should be True if the user wants to save the lookup table (especially if it takes a while to make).
--The parameter path is the path for saving the lookup table (if save is True).  If '', it is saved in the current directory.
--The parameter load should be True if the user wants to load in a dictionary in the directory specified by path (assuming the filename scheme used when save is True).
Other parameters, such as sigma of the Gaussians and the step size 
for the lookup table, can be changed after after if __name__ == '__main__'.

By default, this makes 15x15 frames containing a single PSF and uses centroids made 
with centroids between 7.5 and 8.5.'''

one_to_one = False
paper_plot = False
step = 0.1 # pixels
sigma = 0.4 # pixels
path = ''
save = False
load = False

def gauss_fit(data):
    '''Fits a 2D array (data) to a Gaussian and returns the best-fit parameters and the 
     array containing the output of the Gaussian function with the best-fit parameters.'''
    Y = np.arange(0,len(data))
    X = np.arange(0,len(data[0]))
    X, Y = np.meshgrid(X,Y)
    XY = np.vstack((X.ravel(), Y.ravel()))
    init_guess = (200, 5000, len(data[0])/2, len(data)/2, 2, 2)
    ub = [2**16-1, 2**16-1, len(data[0]), len(data), 10, 10]
    lb = [0, 0, 0, 0, 0, 0]
    bounds = (lb, ub)
    popt, pcov = curve_fit(gauss_spot, XY, data.ravel(), bounds=bounds, p0=init_guess, maxfev=1e5, xtol=1e-15)
    gauss_val = gauss_spot(XY, *popt)
    return *popt, gauss_val

def fit_algorithm(full_frame, pkRow, pkCol, sub_frame_size):
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
    
    offset, A2, xc2, yc2, sigmax2, sigmay2, SigEst2 = gauss_fit(image)
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

    return centroid

def gauss_spot(xy, offset, A, x0, y0, sx, sy):
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

def int_gauss(offset, A, x0, y0, sx, sy, x1, x2, y1, y2):
    '''Double integral of Gaussian function with bounds x1, x2 and y1, y2. Used for 
    finding the value for each pixel of a simulated Gaussian.'''
    pix_val = (offset*(x1 - x2)*(y1 - y2) + 
        (A*np.pi*sx*sy*(erf((-x0 + x1)/(np.sqrt(2)*sx)) - erf((-x0 + x2)/(np.sqrt(2)*sx)))*
        (erf((-y0 + y1)/(np.sqrt(2)*sy)) - erf((-y0 + y2)/(np.sqrt(2)*sy))))/2.)
    return pix_val
    
if __name__ == '__main__':

    M = 15 #window of 15x15 surrounding the PSF
    noise = False # detector noise simulated using emccd_detect
    if one_to_one:
        noise = False
    e_per_dn = 1/.13
    ceil = (2**12 - 6)*e_per_dn #in electrons

    lookup_distance_residuals_list = []
    centroid_distance_residuals_list = []
    
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
                emccd = EMCCDDetect(em_gain=1, full_well_image=ceil, full_well_serial=1e12, dark_current=50, cic=0, read_noise=9, bias=2000, qe=0.7, cr_rate=0, pixel_pitch=10e-6, eperdn=e_per_dn, nbits=12, numel_gain_register=604, meta_path=None)
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

                pk = np.unravel_index(np.argmax(binned_g2), binned_g2.shape)
                centroid = fit_algorithm(binned_g2, pk[0], pk[1], binned_g2.shape[0]+1)
        
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
                scale = np.random.uniform(0.3,0.9)
                for j in x:
                    for k in y:
                        pix_val = int_gauss(0, ceil*scale, x_pts[i], y_pts[i], sigma, sigma, j-0.5, j+0.5, k-0.5, k+0.5)
                        binned_g2[k,j] += pix_val

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

            pk = np.unravel_index(np.argmax(binned_g2), binned_g2.shape)
            centroid = fit_algorithm(binned_g2, pk[0], pk[1], binned_g2.shape[0]+1)


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

            lookup_distance_residuals_list.append(np.square(np.subtract(est_pos[0], true_pos[0])+np.square(np.subtract(est_pos[1], true_pos[1]))))
            centroid_distance_residuals_list.append(np.square(np.subtract(centroid_pos[0], true_pos[0]))+np.square(np.subtract(centroid_pos[1], true_pos[1])))

    if one_to_one:
        print('Number of estimated positions chosen: ', len(est_pos_list))
        print('Number of unique estimated positions chosen: ', np.unique(np.array(est_pos_list),axis=0).shape[0])
    else:
        est_RMSE = np.sqrt(np.mean(lookup_distance_residuals_list)) 

        cent_RMSE = np.sqrt(np.mean(centroid_distance_residuals_list))
        
        print('Lookup Table RMSE mean is {}'.format(est_RMSE))
        std_est_RMSE = np.sqrt(1/(2*len(lookup_distance_residuals_list)))*est_RMSE #np.std(est_RMSE_list)
        print('Lookup Table RMSE STD is {}'.format(std_est_RMSE))

        print('Fitting Algorithm RMSE mean is {}'.format(cent_RMSE))
        std_cent_RMSE = np.sqrt(1/(2*len(centroid_distance_residuals_list)))*cent_RMSE #np.std(cent_RMSE_list)
        print('Fitting Algorithm RMSE STD is {}'.format(std_cent_RMSE))
    pass
    

    ######################### Results quoted in paper
    #900 samples, with noise
    # sigma=0.2, step=0.1:
    # Estimated RMSE mean is 0.28945673794705223
    # Estimated RMSE STD is 0.00682256074208327
    # Centroiding RMSE mean is 0.48838917894274997
    # Centroiding RMSE STD is 0.011511443342951623

    # sigma=0.4, step=0.1:
    # Estimated RMSE mean is 0.040041136074176864
    # Estimated RMSE STD is 0.0009437786281487918
    # Centroiding RMSE mean is 0.25459843313090585
    # Centroiding RMSE STD is 0.0060009426182111095

    # sigma=0.8, step=0.1:
    # Estimated RMSE mean is 0.02730872470033283
    # Estimated RMSE STD is 0.0006436728140387304
    # Centroiding RMSE mean is 0.08979003068865664
    # Centroiding RMSE STD is 0.0021163713194299106

    # sigma=2, step=0.1:
    # Estimated RMSE mean is 0.03376128502942484
    # Estimated RMSE STD is 0.0007957611195292724
    # Centroiding RMSE mean is 0.018322733335014375
    # Centroiding RMSE STD is 0.0004318709663687156

    #########################
    #900 samples, no noise
    # sigma=0.2, step=0.1:
    # Estimated RMSE mean is 0.03730845462333916
    # Estimated RMSE STD is 0.0008793687086584574
    # Centroiding RMSE mean is 0.060289683543658906
    # Centroiding RMSE STD is 0.0014210414689770738

    # sigma=0.4, step=0.1:
    # Estimated RMSE mean is 0.02478269585668995
    # Estimated RMSE STD is 0.0005841337432116406
    # Centroiding RMSE mean is 0.014090324903547703
    # Centroiding RMSE STD is 0.00033211214294734224

    # sigma=0.8, step=0.1:
    # Estimated RMSE mean is 0.030555992620384307
    # Estimated RMSE STD is 0.0007202116529253282
    # Centroiding RMSE mean is 0.0002448514201583148
    # Centroiding RMSE STD is 5.771203319236697e-06

    # sigma=2, step=0.1:
    # Estimated RMSE mean is 0.0292881891685683
    # Estimated RMSE STD is 0.0006903292389923013
    # Centroiding RMSE mean is 1.288059106840853e-07
    # Centroiding RMSE STD is 3.0359844300541826e-09

    ###########
    # 1-to-1 tests, 100 PSFs, no noise:
    # sigma = 0.2, step=0.1:  87 out of 100 (13 times not 1-to-1)
    # sigma = 0.4, step=0.1:  100 out of 100 (0 times not 1-to-1)
    # sigma = 0.8, step=0.1:  100 out of 100 (0 times not 1-to-1)
    # sigma = 2, step=0.1:  100 out of 100 (0 times not 1-to-1)