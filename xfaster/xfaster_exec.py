import os
import sys
import glob
import inspect
import subprocess as sp
import numpy as np
import time
import warnings
from . import xfaster_class as xfc
from . import batch_tools as bt
from . import parse_tools as pt
from collections import OrderedDict


__all__ = [
    "xfaster_run",
    "xfaster_parse",
    "xfaster_submit",
    "xfaster_dump",
    "xfaster_main",
    "XFasterJobGroup",
]


def xfaster_run(
    # common options
    config="config_example.ini",
    output_root=None,
    output_tag=None,
    verbose="notice",
    debug=False,
    dump_state=False,
    checkpoint=None,
    alm_pixel_weights=False,
    alm_iter=None,
    add_log=False,
    # file options
    data_root="all",
    data_subset="full/*0",
    data_root2=None,
    data_subset2=None,
    # binning options
    lmin=2,
    lmax=500,
    pol=True,
    pol_mask=True,
    tbeb=False,
    bin_width=25,
    weighted_bins=False,
    residual_fit=True,
    bin_width_res=25,
    res_specs=None,
    foreground_fit=False,
    beta_fit=False,
    bin_width_fg=30,
    lmin_fg=None,
    lmax_fg=None,
    # beam and kernel options
    pixwin=True,
    mask_type="rectangle",
    window_lmax=None,
    apply_gcorr=False,
    reload_gcorr=False,
    gcorr_file=None,
    # sim ensemble options
    signal_type="synfast",
    signal_subset="*",
    signal_transfer_type=None,
    foreground_type=None,
    foreground_subset="*",
    foreground_transfer_type=None,
    noise_type="stationary",
    noise_subset="*",
    qb_file_sim=None,
    # shape spectrum options
    signal_spec=None,
    signal_transfer_spec=None,
    foreground_spec=None,
    foreground_transfer_spec=None,
    model_r=None,
    freq_ref=359.7,
    beta_ref=1.54,
    freq_ref_transfer=None,
    beta_ref_transfer=None,
    # data options
    data_type="raw",
    template_type=None,
    template_noise_type=None,
    template_alpha_tags=None,
    template_alpha=None,
    reference_type=None,
    ensemble_mean=False,
    ensemble_median=False,
    sim_data=False,
    sim_data_components=["signal", "noise", "foreground"],
    sim_index_signal=None,
    sim_index_noise=None,
    sim_index_foreground=None,
    sim_index_default=0,
    num_sims=1,
    signal_type_sim=None,
    sim_data_r=None,
    noise_type_sim=None,
    qb_file_data=None,
    foreground_type_sim=None,
    template_type_sim=None,
    template_alpha_tags_sim=None,
    template_alpha_sim=None,
    save_sim_data=False,
    # spectrum estimation options
    multi_map=True,
    bandpower_tag=None,
    converge_criteria=0.005,
    cond_noise=1e-5,
    cond_criteria=5e3,
    iter_max=200,
    save_iters=False,
    return_cls=False,
    qb_only=False,
    fix_bb_transfer=False,
    fix_fg_transfer=False,
    null_first_cmb=False,
    delta_beta_prior=None,
    like_profiles=False,
    like_profile_sigma=3.0,
    like_profile_points=100,
    # likelihood options
    likelihood=False,
    mcmc=True,
    mcmc_walkers=50,
    like_converge_criteria=0.01,
    like_tag=None,
    like_lmin=33,
    like_lmax=250,
    like_r_specs=["EE", "BB"],
    like_template_specs=["EE", "BB", "EB"],
    like_alpha_tags="all",
    alpha_prior=None,
    r_prior=[-np.inf, np.inf],
    res_prior=None,
    like_beam_tags="all",
    beam_prior=None,
):
    """
    Main function for running the XFaster algorithm.

    Arguments
    ---------
    config : str
        Configuration file. If path doesn't exist, assumed
        to be in xfaster/config/<config>
    output_root : str
        Directory in which to store output files
    output_tag : str
        File tag for output files
    verbose : str
        Logging verbosity level.  Can be one of ['critical', 'error', 'warning',
        'notice', 'info', 'debug', all'].
    debug : bool
        Store extra data in output files for debugging.
    dump_state : bool
        Store current state immediately prior to bandpowers checkpoint.
        Useful for debugging.
    checkpoint : str
        If supplied, re-compute all steps of the algorithm from this point
        forward.  Valid checkpoints are {checkpoints}.
    alm_pixel_weights : bool
        If True, set the ``use_pixel_weights`` option to True when computing map
        Alms using ``healpy.map2alm``.  If False, sets the ``use_weights``
        option to True instead.  Note: pixel weights ensure accurate Alm
        computation, but can only be used for analyses where ``lmax < 1.5 *
        nside``.
    alm_iter : int
        If given, set the ``iter`` option to the given value when computing map
        Alms using ``healpy.map2alm``.  Using more iterations improves the
        accuracy of the output Alms.  If not set, uses the default number of
        iterations (3) as defined in healpy.  Ignored if ``alm_pixel_weights``
        is True.
    add_log : bool
        If True, write log output to a file instead of to STDOUT.
        The log will be in ``<output_root>/xfaster-<output_tag>.log``.
        This option is useful for logging to file for jobs that
        are run directly (rather than submitted).
    data_root : str
        Root directory where all input files are stored
    data_subset : str
        The subset of the data maps to include from each data split.  Must be a
        glob-parseable string.  Include multiple tags as a comma-delimited
        sequence enclosed in double quotes.
    data_root2 : str
        Path for second set of maps for null test. If set, XFaster performs a
        null test between ``data_root`` and ``data_root2``.
    data_subset2 : str
        The subset of the data maps to include from each data split for the
        second half of a null split.
    lmin : int
       Minimum ell at which to start the lowest bin of the output spectra.
    lmax : int
        Maximum ell for which to compute spectra.
    pol : bool
        If True, polarization spectra are computed.
    pol_mask : bool
        If True, a separate mask is applied for Q/U maps.
    tbeb : bool
        If True, compute TB/EB spectra.
    bin_width : int or array_like of 6 ints
        Width of each ell bin for each of the six output spectra
        (TT, EE, BB, TE, EB, TB).  EE/BB bins should be the same
        in order to handle mixing correctly.
    weighted_bins : bool
        If True, use an lfac-weighted binning operator to construct Cbls.
        By default, a flat binning operator is used.
    residual_fit : bool
        If True, include noise residual bins in the estimator.
    bin_width_res : int or array_like of ints
        Width of each residual spectrum bin.  If a scalar, the same width is
        applied to all spectra for all cross spectra.  Otherwise, must be a list
        of up to nspec * nmaps elements, listing bin widths for each of the
        spectra in ``res_specs`` in order, then ordered by map.
    res_specs : list of strings
        Spectra to include in noise residual fitting.  List values can be any of
        the cross spectra TT, EE, BB, TE, EB, TB, or EEBB for fitting EE and BB
        residuals simultaneously.  If not supplied, this defaults to EEBB for
        polarized maps, or TT for unpolarized maps.
    foreground_fit : bool
        Include foreground bandpowers in the estimator.  If ``foreground_type``
        is set, uses the foreground model in the appropriate sims directory, and
        includes a separate transfer function calculation for the foreground
        components of the covariance.  Otherwise, uses a simple power-law model
        with the same transfer function as the main signal component.
    beta_fit : bool
        If True, include a fit for ``delta_beta`` in the estimator.  Otherwise,
        only fit for foreground amplitudes.
    bin_width_fg : int or array_like of 6 ints
        Width of each ell bin for each of the six output foreground spectra (TT,
        EE, BB, TE, EB, TB).  EE/BB bins should be the same in order to handle
        mixing correctly.  Ignored if neither ``foreground_fit`` nor
        ``foreground_type`` is set.
    lmin_fg : int
        Minimum ell to use for defining foreground bins.  If not set, defaults
        to ``lmin``.
    lmax_fg : int
        Maximum ell to use for defining foreground bins.  If not set, defaults
        to ``lmax``.
    pixwin : bool
        If True, apply pixel window functions to beam windows.
    mask_type : str
        The variant of mask to use
    window_lmax : int
        The size of the window used in computing the mask kernels
    apply_gcorr : bool
        If True, a correction factor is applied to the g (mode counting)
        matrix.  The correction factor should have been pre-computed
        for each map tag.
    reload_gcorr : bool
        If True, reload the gcorr file from the masks directory. Useful when
        iteratively solving for the correction terms.
    gcorr_file : str
        If not None, path to gcorr file. Otherwise, use file labeled
        mask_map_<tag>_gcorr.npy in mask directory for signal, or
        mask_map_<tag>_gcorr_null.npy for nulls.
    signal_type : str
        The variant of signal sims to use for the signal component of the
        covariance model.
    signal_subset : str
        The subset of the signal sims to include.  Must be a glob-parseable
        string.
    signal_transfer_type : str
        The variant of signal sims to use for computing the transfer function.
        If not set, defaults to ``signal_type``.
    noise_type : str
        The variant of noise sims to use for the noise component of the
        covariance model.
    noise_subset : str
        The subset of the noise sims to include.  Must be a glob-parseable
        string.
    foreground_type : str
        The variant of foreground sims to use for the foreground component of
        the covariance model.  If not supplied, and ``foreground_fit`` is False,
        no foreground component is included.
    foreground_subset : str
        The subset of the foreground sims to include.  Must be a glob-parseable
        string.
    foreground_transfer_type : string
        The variant of foreground sims to use for the transfer function for a
        foreground component.  If not set, defaults to ``foreground_type``.
    qb_file_sim : str
        If not None, pointer to a bandpowers.npz file in the output directory,
        to correct the noise ensemble by an appropriate set of residual ``qb``
        values.
    signal_spec : str
        The spectrum data file to use for estimating signal component
        bandpowers.  If not supplied, will search for
        ``spec_signal_<signal_type>.dat`` in the signal sim directory.
    signal_transfer_spec : str
        The spectrum data file used to generate signal sims for computing the
        signal transfer function.  If not supplied, will search for
        ``spec_signal_<signal_transfer_type>.dat`` in the transfer signal sim
        directory.
    foreground_spec : string
        The spectrum data file to use for estimating foreground component
        bandpowers.  If not supplied, will search for
        ``spec_foreground_<foreground_type>.dat`` in the foreground sim
        directory.  Otherwise, use a simple power law model for dust
        foregrounds.
    foreground_transfer_spec : string
        The spectrum data file used to generate foreground sims for computing
        the foreground transfer function.  If not supplied, will search for
        ``spec_foreground_<foreground_transfer_type>.dat`` in the foreground
        transfer sim directory.
    model_r : float
        The ``r`` value to use to compute a spectrum for estimating bandpowers.
        Overrides ``signal_spec``.
    freq_ref : float
        In GHz, reference frequency for dust model. Dust bandpowers output
        will be at this reference frequency.
    beta_ref : float
        The spectral index of the dust model. This is a fixed value, with
        an additive deviation from this value fit for in foreground fitting
        mode.
    freq_ref_transfer : float
        Reference frequency for the dust model used for computing the foreground
        transfer function.  Defaults to ``freq_ref`` if not given.
    beta_ref_transfer : float
        Reference spectral index for the dust model used for computing the
        foreground transfer function.  Defaults to ``beta_ref`` if not given.
    data_type : str
        The data type to use
    template_type : str
        Tag for directory (templates_<template_type>) containing templates
        (e.g. a foreground model) to be scaled by a scalar value per
        map tag and subtracted from the data. The directory is assumed
        to contain reference1 and reference2 subdirectories, each
        containing one template per map tag.
    template_noise_type : string
        Tag for directory containing template noise sims to be averaged and
        scaled similarly to the templates themselves.  These averaged sims are
        used to debias template cross spectra due to correlations in the way the
        noise ensembles are constructed.  Typically, this would be a noise model
        based on the Planck FFP10 ensemble for each half-mission foreground
        template.
    template_alpha_tags : list of strings
        List of map tags from which foreground template maps should be
        subtracted.  These should be the original map tags, not
        those generated for chunk sets.
    template_alpha : list of floats
        Scalar to be applied to template map for subtraction from each of the
        data with tags in the list ``template_alpha_tags``.
    reference_type : string
        Tag for directory containing reobserved reference signals, to be
        subtracted from each data map.  The reference signal maps should be two
        datasets with uncorrelated noise, such as Planck half-mission maps.
        This option is used for removing expected signal residuals from null
        tests.
    ensemble_mean : bool
        If True, substitute S+N ensemble means for Cls to test for bias
        in the estimator.
    ensemble_median : bool
        If True, substitute S+N ensemble median for Cls to test for bias
        in the estimator.
    sim_data : bool
        If True, construct simulated data spectra using the options below.
    sim_data_components : list of strings
        List of components to include in simulated data.  May include signal,
        noise, foreground or template components.
    sim_index_signal : int
        Sim index to use for the signal component that is included in
        ``sim_data_components``.  If None or < 0, takes the value of
        ``sim_index_default``.
    sim_index_noise : int
        Sim index to use for the noise component that is included in
        ``sim_data_components``.  If None or < 0, takes the value of
        ``sim_index_default``.
    sim_index_foreground : int
        Sim index to use for the foreground component that is included in
        ``sim_data_components``.  If None or < 0, takes the value of
        ``sim_index_default``.
    sim_index_default : int
        Default sim index to use for any component with index < 0 or None
        in ``sim_index_<comp>``.
    num_sims : int
        If > 1, repeat the data, bandpowers and likelihood checkpoints this many
        times, incrementing the value of ``sim_index_default`` by 1 each time,
        starting from the input value.  Only used if ``sim_data`` is True.
        All other options remain the same for each iteration.
    signal_type_sim : str
        The variant of signal sims to use for sim_index data maps.
        This enables having a different noise sim ensemble to use for
        sim_index run than the ensemble from which the signal is computed.
    sim_data_r : float
        If not None, construct the signal component of the simulated data by
        selecting the appropriate index from an ensemble of scalar and tensor
        maps, such that the signal component is ``scalar + r * tensor``.  This
        assumes that the tensor simulations are constructed with ``nt=0``, so
        that the linear relationship holds.
    noise_type_sim : str
        The variant of noise sims to use for sim_index fake data map.
        This enables having a different noise sim ensemble to use for
        sim_index run than the ensemble from which the noise is computed.
    qb_file_data : str
        If not None, pointer to a bandpowers.npz file in the output directory,
        to correct the noise component of the simulated data by an appropriate
        set of residual ``qb`` values.
    foreground_type_sim : str
        Tag for directory (foreground_<foreground_type_sim>) where foreground
        sims are stored that should be added to the signal and noise sims
        when running in sim_index mode.
    template_type_sim : string
        Tag for directory containing foreground templates, to be scaled by a
        scalar value per map tag and added to the simulated data.  The directory
        contains one template per map tag.
    template_alpha_tags_sim : list of str
        List of map tags to which foreground template maps should be added, if
        the template component is included in ``sim_data_components``.  These
        should be the original map tags, not those generated for chunk sets.
        If None, use the same values as `template_alpha_tags`.
    template_alpha_sim : list of floats
        Scalar to be applied to template map for addition to each of the
        simulated data maps with tags in the list ``template_alpha_tags``.
        If None, use the same values as `template_alpha`.
    save_sim_data : bool
        If True, save data_xcorr file to disk for simulated data.
    multi_map : bool
        If True, compute all cross-spectra between maps
    bandpower_tag : str
        Tag to append to bandpowers output file
    converge_criteria : float
        The maximum fractional change in qb to signal convergence and
        end iteration
    cond_noise : float
        The level of regularizing noise to add to EE and BB diagonals.
    cond_criteria : float
        Threshold on covariance condition number. Above this, regularizing noise
        will be added to covariance to condition it.
    iter_max : int
        The maximum number of iterations
    save_iters : bool
        If True, store the output of each Fisher iteration, in addition to
        the end result.
    return_cls : bool
        If True, return C_l spectrum rather than the D_l spectrum
    qb_only : bool
        If True, do not compute signal window functions or C_l or D_l bandpowers
    fix_bb_transfer : bool
        If True, after transfer functions have been calculated, impose that the
        BB transfer function is exactly equal to the EE transfer function.
    fix_fg_transfer : bool
        If True, the transfer function for the foreground component is fixed to
        an interpolation of the CMB transfer function onto the foreground
        binning.
    null_first_cmb : bool
        If True, keep first CMB bandpowers fixed to input shape (qb=1).
    delta_beta_prior : float
        The width of the prior on the additive change from beta_ref. If you
        don't want the code to fit for a spectral index different
        from beta_ref, set this to be None.
    like_profiles : bool
        If True, compute profile likelihoods for each qb, leaving all
        others fixed at their maximum likelihood values.  Profiles are
        computed over a range +/--sigma as estimated from the diagonals
        of the inverse Fisher matrix.
    like_profile_sigma : float
        Range in units of 1sigma over which to compute profile likelihoods
    like_profile_points : int
        Number of points to sample along the likelihood profile
    likelihood : bool
        If True, compute the parameter likelihood
    mcmc : bool
        If True, sample the parameter likelihood using an MCMC sampler
    mcmc_walkers : int
        Number of MCMC walkers to use in the likelihood
    like_converge_criteria : float
        Convergence criteria for likelihood MCMC chains
    like_tag : str
        Tag to append to likelihood output file
    like_lmin : int
        The minimum ell value to be included in the likelihood calculation
    like_lmax : int
        The maximum ell value to be included in the likelihood calculation
    like_r_specs : list
        Which spectra to use in the r likelihood.
    like_template_specs : list
        Which spectra to use for alpha in the likelihood.
    like_alpha_tags : list of strings
        List of map tags from which foreground template maps should be
        subtracted and fit in the likelihood. If "all", defaults to
        template_alpha_tags.  If None, alpha fitting in the likelihood is
        disabled.
    alpha_prior: list of floats
        Flat prior edges for allowed alpha values in the likelihood.
        Set to None to not fit for alpha values in the likelihood.
    r_prior: list of floats
        Flat prior edges for allowed r values in the likelihood.
    res_prior: list of floats
        Flat prior edges for allowed qb residual values in the likelihood.
        Set to None to not fit for residual qb values in the likelihood.
    like_beam_tags : list of strings
        List of map tags from which beam error fields are read in to be
        fit for in the likelihood.  If "all", then all available map tags
        in the dataset are included.  If None, then beam error fitting
        in the likelihood is disabled.
    beam_prior: list of floats
        Gaussian prior mean and number of strandard deviations for beam error.
        This Gaussian is applied as a prior in fitting for beam error in the
        likelihood. Set to None to not fit for beam error.
    """
    from . import __version__ as version

    all_opts = locals()

    # py3-compatible CPU timer
    cpu_time = getattr(time, "process_time", getattr(time, "clock", time.time))
    cpu_start = cpu_time()
    time_start = time.time()

    if signal_transfer_type is None:
        signal_transfer_type = signal_type
        all_opts["signal_transfer_type"] = signal_transfer_type

    if foreground_transfer_type is None:
        foreground_transfer_type = foreground_type
        all_opts["foreground_transfer_type"] = foreground_transfer_type

    # make sure sims get rerun correctly
    if (
        signal_transfer_type == signal_type
        and foreground_transfer_type == foreground_type
    ):
        # if signal types match, then sims are run before computing the
        # transfer function, so need to set the correct checkpoint to rerun
        if checkpoint == "sims":
            checkpoint = "sims_transfer"
            all_opts["checkpoint"] = checkpoint

    if template_alpha_tags is None:
        template_alpha_tags = []
        template_alpha = []
    if len(template_alpha_tags) != len(template_alpha):
        raise ValueError(
            "template_alpha_tags and template_alpha must be the same length"
        )
    template_alpha = dict(zip(template_alpha_tags, template_alpha))
    all_opts["template_alpha"] = template_alpha
    all_opts.pop("template_alpha_tags")

    if template_alpha_tags_sim is not None or template_alpha_sim is not None:
        if template_alpha_tags_sim is None:
            template_alpha_tags_sim = template_alpha_tags
        if template_alpha_sim is None:
            template_alpha_sim = [template_alpha[k] for k in template_alpha_tags]
        if len(template_alpha_tags_sim) != len(template_alpha_sim):
            raise ValueError(
                "template_alpha_tags_sim and template_alpha_sim must be the same length"
            )
        template_alpha_sim = dict(zip(template_alpha_tags_sim, template_alpha_sim))
    all_opts["template_alpha_sim"] = template_alpha_sim
    all_opts.pop("template_alpha_tags_sim")

    sim_index = {}
    for k in ["default", "signal", "noise", "foreground"]:
        v = all_opts.pop("sim_index_{}".format(k))
        if v is not None and v >= 0:
            sim_index[k] = v
    all_opts["sim_index"] = sim_index

    if like_alpha_tags is None:
        like_alpha_tags = []
    elif len(like_alpha_tags) == 1 and like_alpha_tags[0] == "all":
        like_alpha_tags = "all"
    all_opts["like_alpha_tags"] = like_alpha_tags

    if like_beam_tags is None:
        like_beam_tags = []
    elif len(like_beam_tags) == 1 and like_beam_tags[0] == "all":
        like_beam_tags = "all"
    all_opts["like_beam_tags"] = like_beam_tags

    # initialize config file
    config_vars = xfc.XFasterConfig(all_opts, "XFaster General")

    common_opts = dict(
        config=config,
        output_root=output_root,
        output_tag=output_tag,
        verbose=verbose,
        debug=debug,
        dump_state=dump_state,
        checkpoint=checkpoint,
        alm_pixel_weights=alm_pixel_weights,
        alm_iter=alm_iter,
        add_log=add_log,
    )
    config_vars.update(common_opts, "XFaster Common")
    common_opts.pop("config")
    common_opts.pop("dump_state")

    # initialize class
    X = xfc.XFaster(config, **common_opts)

    # setup options
    file_opts = dict(
        data_root=data_root,
        data_subset=data_subset,
        data_root2=data_root2,
        data_subset2=data_subset2,
    )
    config_vars.update(file_opts, "File Options")

    X.log("Configuring file structure...", "notice")
    file_vars = X.get_files(**file_opts)
    config_vars.update(file_vars, "File Settings")
    # remove ensemble file arrays from config file
    for k, v in file_vars.items():
        if isinstance(v, np.ndarray) and v.ndim > 1:
            config_vars.remove_option("File Settings", k)
        if isinstance(v, OrderedDict):
            config_vars.set("File Settings", k, dict(v))

    # disable residual fitting in single map mode
    if X.num_maps == 1 or not multi_map:
        residual_fit = False

    bin_opts = dict(
        lmin=lmin,
        lmax=lmax,
        pol=pol,
        pol_mask=pol_mask,
        tbeb=tbeb,
        bin_width=bin_width,
        weighted_bins=weighted_bins,
        residual_fit=residual_fit,
        res_specs=res_specs,
        bin_width_res=bin_width_res,
        foreground_fit=foreground_fit,
        beta_fit=beta_fit,
        bin_width_fg=bin_width_fg,
        lmin_fg=lmin_fg,
        lmax_fg=lmax_fg,
    )
    config_vars.update(bin_opts, "Binning Options")

    data_opts = dict(
        data_type=data_type,
        template_type=template_type,
        template_noise_type=template_noise_type,
        template_alpha=template_alpha,
        reference_type=reference_type,
        ensemble_mean=ensemble_mean,
        ensemble_median=ensemble_median,
        sim=sim_data,
        components=None if not sim_data else sim_data_components,
        index=None if not sim_data else sim_index,
        num_sims=None if not sim_data else num_sims,
        signal_type_sim=signal_type_sim if sim_data_r is None else "r",
        r=sim_data_r,
        noise_type_sim=noise_type_sim,
        qb_file=qb_file_data,
        foreground_type_sim=foreground_type_sim,
        template_type_sim=template_type_sim,
        template_alpha_sim=template_alpha_sim,
        save_sim=save_sim_data,
    )
    config_vars.update(data_opts, "Data Construction Options")
    config_vars.remove_option("XFaster General", "sim_data")
    config_vars.remove_option("XFaster General", "sim_data_components")
    config_vars.remove_option("XFaster General", "sim_data_r")
    config_vars.remove_option("XFaster General", "sim_index")
    config_vars.remove_option("XFaster General", "qb_file_data")
    config_vars.remove_option("XFaster General", "save_sim_data")
    data_opts.pop("num_sims")

    kernel_opts = dict(
        pixwin=pixwin,
        mask_type=mask_type,
        window_lmax=window_lmax,
        apply_gcorr=apply_gcorr,
        reload_gcorr=reload_gcorr,
        gcorr_file=gcorr_file,
    )
    config_vars.update(kernel_opts, "Beam and Kernel Options")

    sim_opts = dict(
        signal_type=signal_type,
        signal_transfer_type=signal_transfer_type,
        signal_subset=signal_subset,
        noise_type=noise_type,
        noise_subset=noise_subset,
        foreground_type=foreground_type,
        foreground_transfer_type=foreground_transfer_type,
        foreground_subset=foreground_subset,
        qb_file=qb_file_sim,
    )
    config_vars.update(sim_opts, "Simulation Ensemble Options")
    config_vars.remove_option("XFaster General", "qb_file_sim")
    sim_opts.pop("signal_transfer_type")
    sim_opts.pop("foreground_transfer_type")
    sim_transfer_opts = sim_opts.copy()
    if signal_transfer_type != signal_type:
        sim_transfer_opts["signal_type"] = signal_transfer_type
        sim_transfer_opts.pop("noise_type")
        sim_transfer_opts.pop("noise_subset")
        sim_transfer_opts.pop("qb_file")
    if foreground_transfer_type != foreground_type:
        sim_transfer_opts["foreground_type"] = foreground_transfer_type
        sim_transfer_opts.pop("noise_type", None)
        sim_transfer_opts.pop("noise_subset", None)
        sim_transfer_opts.pop("qb_file", None)

    if freq_ref_transfer is None:
        freq_ref_transfer = freq_ref
    if beta_ref_transfer is None:
        beta_ref_transfer = beta_ref

    shape_opts = dict(
        filename=signal_spec,
        filename_transfer=signal_transfer_spec,
        filename_fg=foreground_spec,
        filename_transfer_fg=foreground_transfer_spec,
        r=model_r,
        freq_ref=freq_ref,
        beta_ref=beta_ref,
        freq_ref_transfer=freq_ref_transfer,
        beta_ref_transfer=beta_ref_transfer,
    )
    config_vars.update(shape_opts, "Shape Spectrum Options")
    config_vars.remove_option("XFaster General", "signal_spec")
    config_vars.remove_option("XFaster General", "signal_transfer_spec")
    config_vars.remove_option("XFaster General", "foreground_spec")
    config_vars.remove_option("XFaster General", "foreground_transfer_spec")
    config_vars.remove_option("XFaster General", "model_r")
    shape_opts.pop("filename_transfer")
    shape_opts.pop("filename_transfer_fg")
    shape_opts.pop("freq_ref_transfer")
    shape_opts.pop("beta_ref_transfer")
    shape_transfer_opts = shape_opts.copy()
    shape_transfer_opts["filename"] = signal_transfer_spec
    shape_transfer_opts["filename_fg"] = foreground_transfer_spec
    shape_transfer_opts["freq_ref"] = freq_ref_transfer
    shape_transfer_opts["beta_ref"] = beta_ref_transfer

    spec_opts = dict(
        multi_map=multi_map,
        converge_criteria=converge_criteria,
        iter_max=iter_max,
        save_iters=save_iters,
        fix_bb_transfer=fix_bb_transfer,
        fix_fg_transfer=fix_fg_transfer,
        delta_beta_prior=delta_beta_prior,
        cond_noise=cond_noise,
        cond_criteria=cond_criteria,
        null_first_cmb=null_first_cmb,
        return_cls=return_cls,
        qb_only=qb_only,
        like_profiles=like_profiles,
        like_profile_sigma=like_profile_sigma,
        like_profile_points=like_profile_points,
        file_tag=bandpower_tag,
    )
    config_vars.update(spec_opts, "Spectrum Estimation Options")
    config_vars.remove_option("XFaster General", "bandpower_tag")
    spec_opts.pop("multi_map")
    bandpwr_opts = spec_opts.copy()
    bandpwr_opts.pop("fix_bb_transfer")
    bandpwr_opts.pop("fix_fg_transfer")
    spec_opts.pop("file_tag")

    transfer_opts = spec_opts.copy()
    transfer_opts.pop("cond_noise")
    transfer_opts.pop("cond_criteria")
    transfer_opts.pop("delta_beta_prior")
    transfer_opts.pop("null_first_cmb")
    transfer_opts.pop("return_cls")
    transfer_opts.pop("qb_only")
    transfer_opts.pop("like_profiles")
    transfer_opts.pop("like_profile_sigma")
    transfer_opts.pop("like_profile_points")

    like_opts = dict(
        likelihood=likelihood,
        mcmc=mcmc,
        lmin=like_lmin,
        lmax=like_lmax,
        r_specs=like_r_specs,
        template_specs=like_template_specs,
        null_first_cmb=null_first_cmb,
        alpha_tags=like_alpha_tags,
        alpha_prior=alpha_prior,
        r_prior=r_prior,
        res_prior=res_prior,
        beam_tags=like_beam_tags,
        beam_prior=beam_prior,
        num_walkers=mcmc_walkers,
        converge_criteria=like_converge_criteria,
        file_tag=like_tag,
    )
    config_vars.update(like_opts, "Likelihood Estimation Options")
    config_vars.remove_option("XFaster General", "like_lmin")
    config_vars.remove_option("XFaster General", "like_lmax")
    config_vars.remove_option("XFaster General", "like_r_specs")
    config_vars.remove_option("XFaster General", "like_template_specs")
    config_vars.remove_option("XFaster General", "mcmc_walkers")
    config_vars.remove_option("XFaster General", "like_converge_criteria")
    config_vars.remove_option("XFaster General", "like_tag")
    config_vars.remove_option("XFaster General", "like_alpha_tags")
    config_vars.remove_option("XFaster General", "like_beam_tags")
    like_opts.pop("likelihood")

    # store config
    X.save_config(config_vars)

    X.log("Setting up bin definitions...", "notice")
    X.get_bin_def(**bin_opts)

    X.log("Computing mask cross-spectra and weights...", "notice")
    X.get_mask_weights(
        mask_type=mask_type,
        apply_gcorr=apply_gcorr,
        reload_gcorr=reload_gcorr,
        gcorr_file=gcorr_file,
    )

    X.log("Computing kernels...", "notice")
    X.get_kernels(window_lmax=window_lmax)

    X.log("Computing beam window functions...", "notice")
    X.get_beams(pixwin=pixwin)

    X.log("Computing sim ensemble averages for transfer function...", "notice")
    X.get_masked_sims(transfer=True, **sim_transfer_opts)

    X.log("Loading spectrum shape for transfer function...", "notice")
    X.get_signal_shape(transfer=True, **shape_transfer_opts)

    X.log("Computing transfer functions...", "notice")
    X.get_transfer(**transfer_opts)

    X.log("Computing sim ensemble averages...", "notice")
    X.get_masked_sims(**sim_opts)

    if X.null_run:
        X.log("Loading flat spectrum for null test...", "notice")
        X.get_signal_shape(flat=True)
    else:
        X.log("Loading spectrum shape for bandpowers...", "notice")
        X.get_signal_shape(**shape_opts)

    if sim_data:
        idx0 = data_opts["index"]["default"]
    else:
        num_sims = 1
        idx0 = 0

    if dump_state:
        X.save_state(str(int(time_start)))

    bperr = False

    for idx in range(idx0, idx0 + num_sims):
        if sim_data:
            data_opts["index"]["default"] = idx
        X.log(
            "Computing masked {} cross-spectra...".format(
                "simulated data index {}".format(idx) if sim_data else "data"
            ),
            "notice",
        )
        X.get_masked_data(**data_opts)

        if multi_map:
            X.log("Computing multi-map bandpowers...", "notice")
            try:
                qb, inv_fish = X.get_bandpowers(return_qb=True, **bandpwr_opts)
            except RuntimeError:
                bperr = True
                continue

            if likelihood:
                X.log("Computing multi-map likelihood...", "notice")
                X.get_likelihood(qb, inv_fish, **like_opts)

        else:
            for map_tag, map_file in zip(X.map_tags, X.map_files):
                X.log("Processing map {}: {}".format(map_tag, map_file), "notice")

                X.log("Computing bandpowers for map {}".format(map_tag), "info")
                try:
                    qb, inv_fish = X.get_bandpowers(
                        map_tag=map_tag, return_qb=True, **bandpwr_opts
                    )
                except RuntimeError:
                    bperr = True
                    continue

                if likelihood:
                    X.log("Computing likelihoods for map {}".format(map_tag), "info")
                    X.get_likelihood(qb, inv_fish, map_tag=map_tag, **like_opts)

    cpu_elapsed = cpu_time() - cpu_start
    time_elapsed = time.time() - time_start
    X.log(
        "Wall time: {:.2f} s, CPU time: {:.2f} s".format(time_elapsed, cpu_elapsed),
        "notice",
    )

    if bperr:
        raise RuntimeError("Caught error(s) computing bandpowers, check logs.")


xfaster_run.__doc__ = xfaster_run.__doc__.format(checkpoints=xfc.XFaster.checkpoints)


def get_func_defaults(func):
    """
    Return a dictionary containing the default values for each keyword
    argument of the given function

    Arguments
    ---------
    func : function or callable
        This function's keyword arguments will be extracted.

    Returns
    -------
    dict of kwargs and their default values
    """
    pars = inspect.signature(func).parameters
    from collections import OrderedDict

    return OrderedDict(
        [(k, p.default) for k, p in pars.items() if p.default != p.empty]
    )


def extract_func_kwargs(func, kwargs, pop=False, others_ok=True, warn=False):
    """
    Extract arguments for a given function from a kwargs dictionary

    Arguments
    ---------
    func : function or callable
        This function's keyword arguments will be extracted.
    kwargs : dict
        Dictionary of keyword arguments from which to extract.
        NOTE: pass the ``kwargs`` dict itself, not ``**kwargs``
    pop : bool, optional
        Whether to pop matching arguments from kwargs.
    others_ok : bool
        If False, an exception will be raised when kwargs contains keys
        that are not keyword arguments of func.
    warn : bool
        If True, a warning is issued when kwargs contains keys that are not
        keyword arguments of func.  Use with ``others_ok=True``.

    Returns
    -------
    kwargs : dict
        Dict of items from kwargs for which func has matching keyword arguments
    """
    pars = inspect.signature(func).parameters
    pars = [k for k, p in pars.items() if p.default != p.empty]
    ret = {}
    for k in list(kwargs.keys()):
        if k in pars:
            if pop:
                ret[k] = kwargs.pop(k)
            else:
                ret[k] = kwargs.get(k)
        elif not others_ok:
            msg = "Found invalid keyword argument: {}".format(k)
            raise TypeError(msg)
    if warn and kwargs:
        s = ", ".join(kwargs.keys())
        warn("Ignoring invalid keyword arguments: {}".format(s), Warning)
    return ret


def xfaster_parse(args=None, test=False):
    """
    Return a parsed dictionary of arguments for the xfaster execution script.

    Arguments
    ---------
    args : list of strings, optional
        If not supplied, read from the command line (sys.argv) by argparse.
    test : bool, optional
        If True, raise a RuntimeError instead of exiting.  Useful for
        interactive testing.

    Returns
    -------
    args : dict
        Dictionary of parsed options
    """

    import argparse as ap
    from textwrap import dedent
    from . import __version__ as version

    parser_opts = dict(
        description="Run the XFaster algorithm",
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
    )

    # initialize parser
    if test:

        class TestParser(ap.ArgumentParser):
            def __init__(self, *args, **kwargs):
                super(TestParser, self).__init__(*args, **kwargs)

            def error(self, msg):
                self.print_usage(sys.stderr)
                raise RuntimeError(msg)

            def exit(self, status=0, msg=None):
                msg = "exiting with status {}{}".format(
                    status, ": {}".format(msg) if msg else ""
                )
                raise RuntimeError(msg)

        P = TestParser(**parser_opts)
    else:
        P = ap.ArgumentParser(**parser_opts)

    # add --version option
    P.add_argument("--version", action="version", version="%(prog)s " + version)

    # get default argument values from xfaster_run
    defaults = get_func_defaults(xfaster_run)
    defaults.pop("add_log", None)
    rem_args = list(defaults)

    # argument docstrings
    docstr = dedent(xfaster_run.__doc__).split("\n---------\n")[1]
    arg_docs = {}
    arg = None
    argdoc = ""
    for line in docstr.split("\n"):
        if line.strip() and line.startswith("    "):
            if argdoc:
                argdoc += " "
            argdoc += line.strip()
        elif ":" in line or not line.strip():
            if arg:
                if defaults.get(arg, None) in [True, False]:
                    if argdoc.startswith("If True, "):
                        argdoc = argdoc.replace(
                            "If True, ",
                            "If {}set, ".format("not " if defaults[arg] else ""),
                        )
                    if argdoc.startswith("If False, "):
                        argdoc = argdoc.replace(
                            "If False, ",
                            "If {}set, ".format("not " if not defaults[arg] else ""),
                        )
                arg_docs[arg] = argdoc
            arg = line.split(":")[0].strip()
            argdoc = ""

    def add_arg(
        P,
        name,
        argtype=None,
        default=None,
        short=None,
        help=None,
        positional=False,
        **kwargs
    ):
        """
        Helper function for populating command line arguments. Wrapper
        for ArgumentParser.add_argument.

        Arguments
        ---------
        P : argument parser instance
        name : str
            Name of argument to add
        argtype : str
            Data type of argument
        default : arb
            Default value of the argument
        short : str
            Shortened name of argument
        help : str
            Description of argument
        """

        if help is None:
            help = arg_docs.get(name, None)

        name = name.replace("-", "_")
        if positional:
            argname = name
        else:
            argname = "--{}".format(name.replace("_", "-"))
        altname = kwargs.pop("altname", None)

        if default is None:
            default = defaults.get(name, None)
        if name in rem_args:
            rem_args.remove(name)

        if help is None:
            raise ValueError("Missing help text for argument {}".format(name))

        opts = dict(default=default, help=help, action="store")
        if not positional:
            opts["dest"] = name
        opts.update(**kwargs)

        if default is True:
            argname = "--no-{}".format(name.replace("_", "-"))
            opts["action"] = "store_false"
        elif default is False:
            opts["action"] = "store_true"
        else:
            if opts["action"] not in ["store_true", "store_false"]:
                if argtype is None:
                    if isinstance(default, (int, float)):
                        argtype = type(default)
                opts["type"] = argtype

        argnames = (argname,)
        if short is not None:
            if not short.startswith("-"):
                short = "-{}".format(short)
            argnames += (short,)
        if altname is not None:
            argnames += ("--{}".format(altname.replace("_", "-")),)

        P.add_argument(*argnames, **opts)

    # subparsers
    S = P.add_subparsers(
        dest="mode",
        metavar="MODE",
        title="subcommands",
        help="Function to perform. For more help, call: %(prog)s %(metavar)s -h",
    )
    parser_opts.pop("description")

    # populate subparsers
    for mode, helptext in [
        ("run", "run xfaster"),
        ("submit", "submit xfaster job"),
        ("dump", "dump archive from xfaster job to stdout"),
        ("diff", "compare two archive files"),
    ]:
        PP = S.add_parser(mode, help=helptext, **parser_opts)

        if mode == "dump":
            E = PP.add_mutually_exclusive_group(required=True)
            add_arg(E, "output_file", short="-f", help="Archive filename to print")
            add_arg(E, "output_root", short="-r")
            add_arg(PP, "output_tag", short="-t")
            add_arg(
                PP,
                "checkpoint",
                short="-c",
                choices=xfc.XFaster.checkpoints,
                metavar="CHECKPOINT",
                help="Print all files for this checkpoint",
            )
            add_arg(
                PP,
                "keys",
                short="-k",
                nargs="+",
                help="Select keys to print from each file",
            )
            add_arg(
                PP,
                "verbose",
                short="-v",
                default=False,
                action="store_true",
                help="Print complete entries",
            )
            continue

        if mode == "diff":
            add_arg(PP, "file1", positional=True, help="File to compare")
            add_arg(PP, "file2", positional=True, help="File to compare")
            add_arg(
                PP,
                "keys",
                short="-k",
                nargs="+",
                help="Select keys to print from each file",
            )
            add_arg(
                PP,
                "verbose",
                short="-v",
                default=False,
                action="store_true",
                help="Print status of all entries",
            )
            continue

        # common options
        G = PP.add_argument_group("common options")
        add_arg(G, "config", required=True)
        add_arg(G, "output_root", default=os.getcwd())
        add_arg(G, "output_tag")
        add_arg(
            G,
            "verbose",
            short="-v",
            choices=["critical", "error", "warning", "notice", "info", "debug", "all"],
            metavar="LEVEL",
        )
        add_arg(G, "debug")
        add_arg(G, "dump_state")
        add_arg(
            G,
            "checkpoint",
            short="-c",
            choices=xfc.XFaster.checkpoints,
            metavar="CHECKPOINT",
        )
        E = G.add_mutually_exclusive_group()
        add_arg(E, "alm_pixel_weights")
        add_arg(E, "alm_iter", argtype=int)

        # file options
        G = PP.add_argument_group("file options")
        add_arg(G, "data_root", required=True)
        add_arg(G, "data_subset")
        add_arg(G, "data_root2")
        add_arg(G, "data_subset2")

        # binning options
        G = PP.add_argument_group("binning options")
        add_arg(G, "lmin")
        add_arg(G, "lmax")
        add_arg(G, "pol", help="Ignore polarization")
        add_arg(G, "pol_mask", help="Use the same mask for Q/U maps as for T maps")
        add_arg(G, "tbeb")
        add_arg(G, "bin_width", nargs="+")
        add_arg(G, "weighted_bins")
        add_arg(G, "residual_fit")
        add_arg(G, "bin_width_res", nargs="+")
        add_arg(
            G,
            "res_specs",
            nargs="+",
            choices=["TT", "EE", "BB", "TE", "EB", "TB", "EEBB"],
            metavar="SPEC",
        )
        add_arg(G, "foreground_fit")
        add_arg(G, "beta_fit")
        add_arg(G, "bin_width_fg", nargs="+")
        add_arg(G, "lmin_fg", argtype=int)
        add_arg(G, "lmax_fg", argtype=int)

        # beam and kernel options
        G = PP.add_argument_group("beam and kernel options")
        add_arg(G, "pixwin")
        add_arg(G, "mask_type")
        add_arg(G, "window_lmax")
        add_arg(G, "apply_gcorr")
        add_arg(G, "reload_gcorr")
        add_arg(G, "gcorr_file")

        # sim ensemble options
        G = PP.add_argument_group("sim ensemble options")
        add_arg(G, "signal_type")
        add_arg(G, "signal_subset")
        add_arg(G, "signal_transfer_type")
        add_arg(G, "noise_type")
        add_arg(G, "noise_subset")
        add_arg(G, "foreground_type")
        add_arg(G, "foreground_subset")
        add_arg(G, "foreground_transfer_type")
        add_arg(G, "qb_file_sim")

        # shape spectrum options
        G = PP.add_argument_group("shape spectrum options")
        E = G.add_mutually_exclusive_group()
        add_arg(E, "signal_spec")
        add_arg(E, "model_r")
        add_arg(G, "signal_transfer_spec")
        add_arg(G, "foreground_spec")
        add_arg(G, "foreground_transfer_spec")
        add_arg(G, "freq_ref", argtype=float)
        add_arg(G, "beta_ref", argtype=float)
        add_arg(G, "freq_ref_transfer", argtype=float)
        add_arg(G, "beta_ref_transfer", argtype=float)

        # data options
        G = PP.add_argument_group("data options")
        add_arg(G, "data_type")
        add_arg(G, "template_type")
        add_arg(G, "template_noise_type")
        add_arg(G, "template_alpha_tags", nargs="+", metavar="TAG")
        add_arg(G, "template_alpha", nargs="+", argtype=float)
        add_arg(G, "reference_type")
        E = G.add_mutually_exclusive_group()
        add_arg(E, "ensemble_mean")
        add_arg(E, "ensemble_median")
        add_arg(G, "sim_data")
        add_arg(
            G,
            "sim_data_components",
            nargs="+",
            choices=["signal", "noise", "foreground", "template"],
            metavar="COMP",
        )
        add_arg(G, "sim_index_default", argtype=int)
        add_arg(G, "num_sims", argtype=int)
        add_arg(G, "sim_index_signal", argtype=int)
        add_arg(G, "sim_index_noise", argtype=int)
        add_arg(G, "sim_index_foreground", argtype=int)
        add_arg(G, "signal_type_sim")
        add_arg(G, "sim_data_r", argtype=float)
        add_arg(G, "noise_type_sim")
        add_arg(G, "qb_file_data")
        add_arg(G, "foreground_type_sim")
        add_arg(G, "template_type_sim")
        add_arg(G, "template_alpha_tags_sim", nargs="+", metavar="TAG")
        add_arg(G, "template_alpha_sim", nargs="+", argtype=float)
        add_arg(G, "save_sim_data")

        # spectrum estimation options
        G = PP.add_argument_group("spectrum estimation options")
        add_arg(G, "multi_map")
        add_arg(G, "bandpower_tag")
        add_arg(G, "converge_criteria")
        add_arg(G, "cond_noise", argtype=float)
        add_arg(G, "cond_criteria", argtype=float)
        add_arg(G, "iter_max", argtype=int)
        add_arg(G, "save_iters")
        add_arg(G, "return_cls")
        add_arg(G, "qb_only")
        add_arg(G, "fix_bb_transfer")
        add_arg(G, "fix_fg_transfer")
        add_arg(G, "null_first_cmb")
        add_arg(G, "delta_beta_prior", argtype=float)
        add_arg(G, "like_profiles")
        add_arg(G, "like_profile_sigma", argtype=float)
        add_arg(G, "like_profile_points", argtype=int)

        G = PP.add_argument_group("likelihood options")
        add_arg(G, "likelihood")
        add_arg(G, "mcmc")
        add_arg(G, "mcmc_walkers")
        add_arg(G, "like_converge_criteria", argtype=float)
        add_arg(G, "like_tag")
        add_arg(G, "like_lmin", argtype=int)
        add_arg(G, "like_lmax", argtype=int)
        add_arg(
            G,
            "like_r_specs",
            nargs="+",
            choices=["TT", "EE", "BB", "TE", "EB", "TB"],
            metavar="SPEC",
        )
        add_arg(
            G,
            "like_template_specs",
            nargs="+",
            choices=["TT", "EE", "BB", "TE", "EB", "TB"],
            metavar="SPEC",
        )
        add_arg(G, "like_alpha_tags", nargs="+", metavar="TAG")
        add_arg(G, "alpha_prior", nargs=2)
        add_arg(G, "r_prior", nargs=2)
        add_arg(G, "res_prior", nargs=2)
        add_arg(G, "like_beam_tags", nargs="+", metavar="TAG")
        add_arg(G, "beam_prior", nargs=2)

        # submit args
        if mode == "submit":
            G = PP.add_argument_group("submit options")
            G.add_argument(
                "--job-prefix",
                action="store",
                help="Name to prefix to all submitted jobs",
            )
            G.add_argument(
                "-q", "--queue", action="store", default=None, help="Queue to submit to"
            )
            G.add_argument(
                "--nodes",
                action="store",
                type=str,
                default="1",
                help="Number of nodes to use. Or node name",
            )
            G.add_argument(
                "--ppn",
                action="store",
                type=int,
                default=8,
                help="Number of processors per node",
            )
            G.add_argument(
                "--mem",
                action="store",
                type=float,
                default=5,
                help="Memory per process, in GB",
            )
            E = G.add_mutually_exclusive_group()
            E.add_argument(
                "--cput",
                action="store",
                default=None,
                type=float,
                help="cput per process in hours",
            )
            E.add_argument(
                "--wallt",
                action="store",
                default=None,
                type=float,
                help="walltime in hours",
            )
            G.add_argument(
                "--nice",
                action="store",
                type=int,
                default=0,
                help="Scheduling priority from -5000 (high) to 5000",
            )
            G.add_argument(
                "--omp-threads",
                action="store",
                type=int,
                default=None,
                help="Number of OMP threads to use",
            )
            G.add_argument(
                "--slurm",
                action="store_true",
                default=False,
                help="Submit a slurm script rather than PBS",
            )
            G.add_argument(
                "--env-script", help="Script to source in jobs to set up environment"
            )
            G.add_argument("--exclude", help="Nodes to exclude")
            G.add_argument(
                "--dep-afterok",
                action="store",
                nargs="+",
                default=None,
                help="List of job IDs to wait on completion of before running job",
            )

        # other arguments
        PP.add_argument(
            "--test",
            action="store_true",
            default=False,
            help="Print options for debugging",
        )

    # check that all xfaster_run arguments have been handled by the parser
    if len(rem_args):
        warnings.warn(
            "Argument(s) {} not handled by the parser".format(rem_args),
            xfc.XFasterWarning,
        )
    # parse arguments
    args = P.parse_args(args=args)

    # default mode, required for python 3.7 or newer
    if args.mode is None:
        P.error("the following arguments are required: MODE")

    # fix arguments meant to be empty
    for k, v in vars(args).items():
        if str(v).lower().strip() == "none":
            setattr(args, k, None)
        elif not np.isscalar(v) and len(v) == 1:
            v = v[0]

    # test mode
    if args.mode not in ["submit", "dump", "diff"]:
        if args.test:
            msg = ",\n".join(
                "{}={!r}".format(k, v) for k, v in sorted(vars(args).items())
            )
            P.exit(0, "{}\nargument test\n".format(msg))
        delattr(args, "test")

    # return a dictionary
    return vars(args)


class XFasterJobGroup(object):
    def __init__(self):
        """
        Class for parsing xfaster options into a job script, and optionally
        grouping the jobs together.
        """
        self.reset()

    def reset(self):
        """
        Initialize to a reset state.
        """
        self.output = None
        self.job_list = []
        self.qsub_args = {}

    def add_job(self, **kwargs):
        """
        Add xfaster job to script.

        Keyword arguments
        -----------------
        Most should correspond to arguments accepted by ``xfaster_run``.
        If job-related arguments are present, they will be passed to
        ``set_job_options``.
        """

        # set job options
        job_opts = extract_func_kwargs(
            self.set_job_options, kwargs, pop=True, others_ok=True
        )

        # ensure absolute paths for submit
        for key in ["config", "data_root", "data_root2", "output_root"]:
            value = kwargs.get(key, None)
            if value is not None:
                value = os.path.abspath(value)
                kwargs[key] = value

        output_root = kwargs.get("output_root")
        output_tag = kwargs.get("output_tag")
        if output_root is not None:
            if output_tag is not None:
                output_root = os.path.join(output_root, output_tag)
            job_opts["output"] = output_root

        if job_opts:
            job_prefix = job_opts.get("job_prefix")
            if job_prefix is None:
                job_prefix = "xfaster"
                if output_tag is not None:
                    job_prefix = "_".join([job_prefix, output_tag])
                job_opts["job_prefix"] = job_prefix
            self.set_job_options(**job_opts)

        # construct command
        cmd = "xfaster run".split()

        # figure out variable types from default values
        defaults = get_func_defaults(xfaster_run)
        for a in defaults:
            if a not in kwargs:
                continue

            v = kwargs.pop(a)
            s = a.replace("_", "-")

            da = defaults[a]
            if da is True and v is False:
                cmd += ["--no-{}".format(s)]
            elif da is False and v is True:
                cmd += ["--{}".format(s)]
            elif v != da:
                cmd += ["--{}".format(s)]
                if a in [
                    "template_alpha_tags",
                    "template_alpha",
                    "template_alpha_tags_sim",
                    "template_alpha_sim",
                    "res_specs",
                    "like_mask",
                    "r_prior",
                    "like_alpha_tags",
                    "alpha_prior",
                    "like_beam_tags",
                    "beam_prior",
                    "res_prior",
                    "like_r_specs",
                    "like_template_specs",
                    "sim_data_components",
                    "bin_width",
                    "bin_width_res",
                    "bin_width_fg",
                ]:
                    if np.isscalar(v):
                        v = [v]
                    if "prior" in a:
                        # special case to allow -inf to work
                        if v is None:
                            cmd += ["None"]
                        else:
                            cmd += ["' {}'".format(vv) for vv in v]

                    else:
                        cmd += [str(vv) for vv in v]

                elif a in [
                    "noise_subset",
                    "signal_subset",
                    "foreground_subset",
                    "data_subset",
                    "data_subset2",
                ]:
                    # add quotes around glob parseable args to avoid weird
                    # behavior
                    cmd += ["'{}'".format(v)]

                else:
                    cmd += [str(v)]

        if len(kwargs):
            # Args that are not being parsed-- raise an error
            raise KeyError("{} arguments not recognized.".format(kwargs))

        self.job_list.append(" ".join(cmd))

    def set_job_options(
        self,
        output=None,
        workdir=None,
        cput=None,
        wallt=None,
        ppn=8,
        nodes=1,
        mem=5,
        env_script=None,
        omp_threads=None,
        nice=0,
        queue=None,
        job_prefix=None,
        test=False,
        pbs=False,
        dep_afterok=None,
        exclude=None,
    ):
        """
        Parse options that control the job script, rather than xfaster.
        Passed to batch_tools.batch_sub.

        Arguments
        ---------
        output : string, optional
            Path for output scheduler files, in a logs subdirectory.
            If None, use output_root. Overrided by workdir.
        workdir : string, optional
            If not None, path to output scheduler files. Overrides output.
        cput : string or float or datetime.timedelta, optional
            Amount of CPU time requested.
            String values should be in the format HH:MM:SS, e.g. '10:00:00'.
            Numerical values are interpreted as a number of hours.
        wallt : string or float or datetime.timedelta, optional
            Amount of wall clock time requested.
            String values should be in the format HH:MM:SS, e.g. '10:00:00'.
            Numerical values are interpreted as a number of hours.
        ppn : int, optional
            Numper of processes per node
        nodes : int or string, optional
            Number of nodes to use in job
            If a string, will be passed as-is to PBS -l node= resource
            If using SLURM and a string, will overwrite node_list if None
        mem : float or string, optional
            Amount of memory to request for the job. float values in GB.
            Or pass a string (eg '4gb') to use directly.
        env_script : string, optional
            Path to script to source during job script preamble
            For loading modules, setting environment variables, etc
        omp_threads : int, optional
            Number of OpenMP threads to use per process
        nice : int, optional
            Adjust scheduling priority (SLURM only). Range from -5000 (highest
            priority) to 5000 (lowest priority).
            Note: actual submitted --nice value is 5000 higher, since negative
            values require special privilege.
        queue : string, optional
            The name of the queue to which to submit jobs
        job_prefix : string, optional
            The name of the job. Default: xfaster
        test : bool
            If True, only print out the job submission script, don't submit it.
        pbs : bool
            If True, use pbs scheduler. Else, use slurm.
        dep_afterok : string or list of strings
            Dependency. Job ID (or IDs) on which to wait for successful completion,
            before starting this job
        exclude : string or list of strings
            List of nodes that will be excluded for job. SLURM-only.

        """

        # XFaster runs faster with OMP
        if omp_threads is None:
            omp_threads = ppn

        # default job prefix
        if job_prefix is None:
            job_prefix = "xfaster"

        # create output directories
        if output is None:
            output = self.output
        output = os.path.abspath(output)
        if workdir is None:
            workdir = os.path.join(output, "logs")

        if cput is not None:
            try:
                # split at ":" since extra resource modifiers can go after
                cput *= ppn * int(str(nodes).split(":")[0])
            except ValueError:
                # could not convert nodes to int, assume one node (name)
                cput *= ppn
        if mem is not None:
            mem *= ppn

        scheduler = "slurm"

        self.batch_args = dict(
            workdir=workdir,
            mem=mem,
            nodes=nodes,
            ppn=ppn,
            cput=cput,
            wallt=wallt,
            queue=queue,
            env_script=env_script,
            omp_threads=omp_threads,
            nice=nice,
            delete=False,
            submit=not test,
            debug=test,
            scheduler=scheduler,
            name=job_prefix,
            dep_afterok=dep_afterok,
            exclude=exclude,
        )

    def submit(self, group_by=None, verbose=True, **kwargs):
        """
        Submit jobs that have been added.

        Arguments
        ---------
        group_by : int, optional
            Group xfaster calls into jobs with this many calls each.
        verbose : bool, optional
            Print the working directory, and the job ID if submitted successfully.

        Returns
        -------
        job_ids : list of strings
            The IDs of the submitted jobs
        """
        if not self.job_list:
            raise RuntimeError("No xfaster jobs have been added.")
        if group_by is None:
            group_by = len(self.job_list)
        if kwargs:
            self.set_job_options(**kwargs)
        if not self.batch_args:
            raise RuntimeError("No job options specified")
        job_ids = bt.batch_group(
            self.job_list,
            group_by=group_by,
            serial=True,
            verbose=verbose,
            **self.batch_args
        )
        self.reset()
        return job_ids


def xfaster_submit(**kwargs):
    """
    Submit a single xfaster job. The arguments here should agree exactly
    with the command line flags for submit mode, with kwargs passed to
    ``xfaster_run``. Run ``xfaster --help`` for help.
    """
    xg = XFasterJobGroup()
    xg.add_job(**kwargs)
    return xg.submit(group_by=1, verbose=True)


def xfaster_dump(
    output_file=None,
    output_root=None,
    output_tag=None,
    checkpoint=None,
    keys=None,
    verbose=False,
):
    """
    Print the contents of a set of output archive files.

    Arguments
    ---------
    output_file : str
        Path to an output npz archive file.
    output_root : str
        Path to an xfaster output directory containing npz archive files.
        Ignored if ``output_file`` is set.
    output_tag : str
        Optional file tag.  Ignored if ``output_file`` is set.
    checkpoint : str
        Checkpoint for which to print all matching archive files.  Ignored if
        ``output_file`` is set.
    keys : list of str
        List of keys to print for each matching archive file.
    verbose : bool
        If True, print the entire contents of the dictionary.  Otherwise,
        truncate each entry to a single line.
    """

    assert output_file is not None or output_root is not None

    if output_file is not None:
        files = [output_file]

    else:
        if output_tag is not None:
            output_root = os.path.join(output_root, output_tag)
            output_tag = "_{}".format(output_tag)
        else:
            output_tag = ""

        patterns = {
            "all": "*{}.npz".format(output_tag),
            "files": "files*{}.npz".format(output_tag),
            "masks": "masks_xcorr*{}.npz".format(output_tag),
            "kernels": "kernels*{}.npz".format(output_tag),
            "sims_transfer": "simx_xcorr*{}.npz".format(output_tag),
            "shape_transfer": "shape_transfer*{}.npz".format(output_tag),
            "transfer": "transfer_all*{}.npz".format(output_tag),
            "sims": "sims_xcorr*{}.npz".format(output_tag),
            "beams": "beams{}.npz".format(output_tag),
            "data": "data*xcorr*{}.npz".format(output_tag),
            "sim_data": "sim_data*xcorr*{}.npz".format(output_tag),
            "template_noise": "template_noise*{}.npz".format(output_tag),
            "shape": "shape*{}.npz".format(output_tag),
            "bandpowers": "bandpowers*{}.npz".format(output_tag),
            "beam_errors": "beam_errors*{}.npz".format(output_tag),
            "likelihood": "like_mcmc*{}.npz".format(output_tag),
        }

        if checkpoint is None:
            checkpoint = "all"

        files = sorted(glob.glob(os.path.join(output_root, patterns[checkpoint])))
        if not len(files):
            raise IOError(
                "No files found for checkpoint {} in directory {}".format(
                    checkpoint, output_root
                )
            )

    for f in files:
        data = pt.load_and_parse(f)
        if keys is not None:
            data = {k: data[k] for k in keys}
        print("***** {} *****".format(f))
        print("{")
        for k, v in data.items():
            txt = repr(v)
            if not verbose:
                txt = txt.split("\n")
                if len(txt) == 1:
                    txt = txt[0]
                else:
                    if isinstance(v, dict):
                        txt = repr(v.keys())
                    elif isinstance(v, np.ndarray):
                        if v.ndim == 1 and v.dtype.char == "U":
                            txt = str(list(v))
                        else:
                            txt = "shape {} array of {}".format(v.shape, repr(v.dtype))
                    else:
                        txt = txt[0] + " ..."
            print("    '{}': {},".format(k, txt))
        print("}")


def xfaster_diff(file1, file2, keys=None, verbose=False):
    """
    Compare two output archive files to each other.  Uses setdiff or numpy.diff
    as appropriate.

    Arguments
    ---------
    file1 : str
    file2 : str
        Filenames to compare to each other
    keys : list of str
        List of keys to compare between files
    verbose : bool
        If True, print the status of all keys, even if identical.  Otherwise,
        only print the status for keys that do not match between files.
    """

    def load1(f):
        d = pt.load_and_parse(f)
        if keys is not None:
            d = {k: d[k] for k in keys if k in d}
        return d

    data1, data2 = [load1(f) for f in [file1, file2]]

    def compare(d1, d2, prefix=""):
        if isinstance(d1, (str, int, float)) or d1 is None:
            if (d1 is None and d2 is not None) or (d1 is not None and d1 != d2):
                print("{}{} != {}".format(prefix, d1, d2))
            elif verbose:
                print("{} SAME: {}".format(prefix, d1))

        elif isinstance(d1, (list, np.ndarray, tuple)):
            d1, d2 = [np.asarray(x) for x in [d1, d2]]
            if d1.dtype.kind in ["U", "S"]:
                s1 = set(d1.ravel())
                s2 = set(d2.ravel())
                sd = s1 - s2
                if sd:
                    print("{}only in {}: {}".format(prefix, file1, sd))
                sd = s2 - s1
                if sd:
                    print("{}only in {}: {}".format(prefix, file2, sd))
                if verbose and not s1 ^ s2:
                    print("{}SAME".format(prefix))

            elif not np.allclose(d1, d2):
                print("{}{}".format(prefix, d1 - d2))

            elif verbose:
                print("{}ALL CLOSE".format(prefix))

        elif isinstance(d1, dict):
            all_keys = sorted(set(d1) | set(d2))
            for k in all_keys:
                if k not in d1:
                    print("{}{}: only in {}".format(prefix, k, file2))
                elif k not in d2:
                    print("{}{}: only in {}".format(prefix, k, file1))
                else:
                    compare(d1[k], d2[k], "{}{}: ".format(prefix, k))

        else:
            print(
                "{}: parser error: A: {} {} B: {} {}".format(
                    prefix, type(d1), d1, type(d2), d2
                )
            )

    compare(data1, data2)


def xfaster_main():
    """
    Main entry point for command-line interface.
    """
    # parse arguments
    args = xfaster_parse()
    mode = args.pop("mode")

    if mode == "submit":
        # submit a job
        xfaster_submit(**args)

    elif mode == "run":
        # run the analysis
        xfaster_run(**args)

    elif mode == "dump":
        # dump archive files to stdout
        xfaster_dump(**args)

    elif mode == "diff":
        # diff archive files
        xfaster_diff(**args)
