## Core information to acquire lab data and data formats


### Data Acquisition

The basic procedure is as follows (as demonstrated fully in `example_perform_lab_measurements_essential.py`), and uses methods from `pllab.py`. 

#### See `example_perform_lab_measurements_essential.py` and corresponding methods in `pllab.py` for details

1. Load a cube of SLM patterns using   `pllab.load_slmims()`


2. Define a set of desired source2 ('planet') positions and contrasts in  
`src2params = np.array([xposns, yposns, contrasts])`


3. Run the measurement procedure, which will show each SLM pattern, set source2 to the required position and contrast, and acquire the camera frames, using  
`all_imdata = pllab.run_measurements_shm(return_data=True, src2params=src2params)`


4. Save the results (and source2 parameters if desired)  
`pllab.savedata(filename=savefilename, savedir=savedatadir)`  
`np.savez(savefilename.npz', src2params=src2params, xposns=xposns, yposns=yposns)`

<br>

---

### Data Formats
#### SLM pattern files
Usually named `slmcube_DATE_DESCRIPTION_fileXX.npz` or similar.  
File contains the following:
* `all_slmims` - array of shape `[n_frames, x_px, y_px]` to be displayed on SLM.  
  * **IMPORTANT:** must be of type `uint8` and map directly to values sent to SLM SDK.
  * If `insert_as_subimage=True` in `pllab.load_slmims()`, `[x_px, y_px]` is the size of active region on SLM, otehrwise it's the size of the whole 1024x1024 SLM.


* `slmloc` - array specifying location and radius of active SLM region, in which `all_slmims` will be placed.
  * Format is `[x_centre, y_centre, radius]`


* `all_slmim_params` - dict of useful parameters describing the contained patterns.
  * For example, for simulated seeing the keys are `r0`, `L0`, `windspeed`, `windangle`, `timespan`, `seeing_global_scaling`, `reset_each_frame`

<br>

#### Acquired data files
Usually named `pllabdata_DATE_DESCRIPTION_SLM-PATTERN-FILE__fileXX.npz`
File contains the following:
* `imcube_cam0`, `imcube_cam1`, ... - cubes of the *RAW* (not dark-subtracted, etc.) acquired camera frames, one cube per camera.  In the current scripts there are cubes for two cameras, with `cam0` being the PSF and `cam1` being the PL output.
  * `imcube_camXX` has shape `[n_frames, x_px, y_px]`, of type `int16`.


* `darkframes` - contains the master darkframes (i.e. assembled from many frames using `pllab.take_darks()` for all cameras.
  * A numpy array of shape `[n_cameras,]` (so `[2,]` for the current case)
  * Each element contains a numpy array of shape `[x_px, y_px]` (corresponding to the size of that camera's images) containing the darkframe, of type `float64`.


* `slmims_filename` - the full filename (including path) of the corresponding SLM pattern file.


* `darkframes` - filename of the darkframe npz file to be used (if applicable)


*  `all_slmim_params` - same as `all_slmim_params` in the SLM pattern files - see above.

<br>

#### Source2 parameter files
Usually named `pllabdata_DATE_DESCRIPTION_SLM-PATTERN-FILE__fileXX_src2params.npz`
File contains the following:
* `src2params` - array of shape `[3,n_frames]`, describing location and contrast of source2 ('planet'), with each slice being `[xposns, yposns, contrasts]`

<br>

---


### Suggested representative example files
A good set of files which represent a typical, basic measurement set would be
* **SLM file:** `slmcube_202400708_seeing_0.4-10-scl1_rand-flatn2_10K_01_file00.npz`  


* **Acquired data file:** `pllabdata_20240708_randsrc2_seeing-rep2_01_slmcube_202400708_seeing_0.4-10-scl1_rand-flatn2_10K_01_file00`  


* **Source2 parameter file:** `pllabdata_20240708_randsrc2_seeing-rep2_01_slmcube_202400708_seeing_0.4-10-scl1_rand-flatn2_10K_01_file00_src2params`

There are 10 files (`_file00`, `_file01`, ...), each containing 10,000 SLM patterns / measurements.