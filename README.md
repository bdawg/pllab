## PL IR testbed - Generate, run experiments and acquire data

This contains the continually modified code for the IR PL testbed. Main features:
- Generate sets of test data to put on the SLM, including sine modes, turbulence and amplitude+intensity patterns using checkerboard method
- Acquire data asynchronously from the4 C-Red 2 cameras, by running them in separate python processes and transferring data using shared memory
- Perform experiments where SLM patterns are displayed and C-Red 2 frames acquired, precisely syncd at maximum speed
- Various test and calibration tasks, including determining beam position on SLM, ...
- Basic plotting and analysis tools


### Important files

| Filename        | Description         |
| ------------- |-------------|
| `pllab.py`      | Main class for performing experiments, including showing SLM pattens and acquiring frames |
| `plcams.py`      | Class handles control and acquisition from C-Red 2 cameras, uses FliSdk_V2|
| `plcam_camprocess.py` | Class + script to asynchronously acquire images from C-Red 2 cameras, run in its own process (one per camera) |
|`plslm.py`| Class to communicate with Meadowlark SLM|
|`example_perform_shm_measurements.py` | Example script showing how to take a measurement set of SLM patterns, plus other useful functions|

See other `example_...` files for examples of other functions

`working_...` files are scripts containing WIP useful data generation and acquisition tasks