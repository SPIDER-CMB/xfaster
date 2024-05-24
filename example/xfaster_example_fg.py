import xfaster as xf

# use this script to run the xfaster algorithm with your own options
# like the example below

submit = False  # submit to queue
submit_opts = {
    "wallt": 5,
    "mem": 3,
    "ppn": 1,
    "omp_threads": 1,
}

# Paths for the following keys are relative to where this script is run:
#     config, data_root, data_root2, output_root
# If running this script from a different directory, make sure these
# keys point to the correct locations.

xfaster_opts = {
    # run options
    "pol": True,
    "pol_mask": True,
    "bin_width": 25,
    "lmax": 500,
    "multi_map": True,
    "likelihood": False,
    "output_root": "outputs_example_fg",
    "output_tag": "95x150",
    # input files
    "config": "config_example.ini",
    "data_root": "maps_example",
    "data_subset": "full/*95,full/*150",
    "data_type": "cmbfg",
    "noise_type": "gaussian",
    "mask_type": "rectangle",
    "signal_type": "synfast",
    "foreground_type": "gaussian",
    "data_root2": None,
    "data_subset2": None,
    # residual fitting
    "residual_fit": True,
    "bin_width_res": 100,
    # foreground fitting
    "foreground_fit": True,
    "beta_fit": False,
    "bin_width_fg": 40,
    "lmin_fg": None,
    "lmax_fg": None,
    "beta_ref": 1.54,
    "freq_ref": 359.7,
    # spectrum
    "ensemble_mean": False,
    "tbeb": True,
    "fix_fg_transfer": True,
    "converge_criteria": 0.005,
    "iter_max": 200,
    "save_iters": False,
    # likelihood
    "like_lmin": 26,
    "like_lmax": 250,
    # beams
    "pixwin": True,
    "verbose": "info",
    "checkpoint": None,
}

if submit:
    xfaster_opts.update(**submit_opts)
    xf.xfaster_submit(**xfaster_opts)
else:
    xf.xfaster_run(**xfaster_opts)
