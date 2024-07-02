# lookup_table
Method of centroiding undersampled PSFs using binned simulated PSFs correlated to simulated position

Reliable centroiding of point spread functions (PSFs) in a star scene is often crucially important for astronomical observations.  Accurate determination of angular distances between stars and pointing directions relies on accurate and consistent centroiding, but this is in general not possible with a simple fit when PSFs are undersampled (sampled at a rate less than what is needed for Nyquist sampling).  When prominent features of a PSF lie on non-integer pixel values, the undersampling of the pixels does not faithfully represent the PSF, and consistent centroiding is not achievable with a simple fit.  One method of achieving better resolution of the PSF is through dithering.  However, there may be cases when dithering is not desired or not possible (for fast tracking of particular targets, when observation and processing time is short, when observations are done in parallel or when the physical constraints of a lab testbed do not allow for dithering).  In this case, if the expected PSF can be modeled well, whether through software or through examination with an interferometer, a lookup table which pairs simulated PSFs with their known simulated centroid positions can provide a useful method of centroiding.  One assigns the centroid position of an observed PSF to the position associated with the PSF in the lookup table that has the smallest squared residual with respect to the observed PSF.

The script is `lookup_table.py`.  It simulates symmetric Gaussian PSFs and demonstrates that this lookup table method is more accurate than a traditional fitting method for undersampled PSFs.  For more details, see the corresponding paper:  (coming soon)

The script utilizes the module `emccd_detect`, which must be installed before using the script.  It is publicly available here:
 https://github.com/wfirst-cgi/emccd\_detect
Simple installation instructions are included. 

![image](https://github.com/kjl0025/lookup_table/assets/90057179/08614308-76fe-46d0-8703-17cb69e33cc4)
