G-Correction Calculation
========================

Code package for calculating correction factors to g_ell using pre-generated
simulation maps for each single mask. This procedure assumes you are running
this code on a cluster-- it is extremely slow to do otherwise.

There is one main command-line utility that calls two functions from the
:py:mod:`~xfaster.gcorr_tools` library:

1. :py:func:`~xfaster.gcorr_tools.xfaster_gcorr_once` -- This function calls
   XFaster to run or submit jobs for gcorr runs.
2. :py:func:`~xfaster.gcorr_tools.process_gcorr` -- A function that computes the
   gcorr correction from the ensemble of bandpowers, updates the running total,
   and backs up the necessary files from each iteration.
3. ``xfaster gcorr`` -- An xfaster command-line utility, that calls function 1
   and 2 to get a new gcorr.npz file each time. This is the main code you'll
   run.

There is also a config file with options specific to computing gcorr.

Procedure
---------

1. Edit the gcorr config file to suit your purposes. Examples are given for
   signal and null runs in the examples directory. Required fields are:

   * ``null`` -- must be true for null tests and false for signal runs
   * ``map_tags`` -- comma-separated list of map tags
   * ``data_subset`` -- the globabble data_subset argument to xfaster_run, but
     without map tags. So, "full", "chunk*", etc.
   * ``output_root`` -- the parent directory where your gcorr XFaster runs will
     be written
   * ``nsim`` -- the number of simulations to use to compute gcorr
   * ``[xfaster_opts]`` -- this is where you'll put any options that will be
     directly input to :py:func:`~xfaster.xfaster_exec.xfaster_run`
   * ``[submit_opts]`` -- this is where you'll put any options that will be
     directly input to :py:func:`~xfaster.xfaster_exec.xfaster_submit`, in
     addition to those in ``[xfaster_opts]``

2. Run ``xfaster gcorr`` once to get the full set of XFaster output files in the
   output directory.  Since we haven't computed gcorr yet, this will set
   ``apply_gcorr=False``. Make sure to use as many OMP threads as possible since
   this is the step where the sims_xcorr file, which benefits the most from
   extra threads, is computed.  Your command should look like this::

      xfaster gcorr path-to-my-gcorr-config.ini

3. Run ``xfaster gcorr`` until convergence is reached. In practice, you will run
   the command above and wait for it to finish. If you include the
   ``--max-iters`` option with a non-zero value, the code will try to determine
   whether convergence or max_iters has been reached and stop on its own.
   Otherwise, you can look at the correction-to-the-correction that it both
   prints and plots (it should converge to 1s for TT, EE, BB), and rerun the
   same command if it hasn't converged.  In much more detail, here's what the
   code does:

   1. Call :py:func:`~xfaster.gcorr_tools.xfaster_gcorr_once` for the 0th sim
      seed while also reloading gcorr (if this is not the first iteration).
      This does a couple things-- saves the new gcorr in the ``masks_xcorr``
      file, so later seeds will use the right thing. And recompute the transfer
      function, which doesn't depend on the ``sim_index``, so is only necessary
      to do once.
   2. After the transfer functions are all on disk, submit individual jobs for
      all the other seeds, just doing the bandpowers step for those.
   3. Once they're all done, run :py:func:`~xfaster.gcorr_tools.compute_gcal`,
      and save a correction-to-gcorr as ``gcorr_corr_<tag>_iter<iter>.npz`` in the
      rundir.
   4. If not the first iteration, load up the correction-to-gcorr computed for
      this iteration. Multiply it by the total gcorr, and save that to the
      output directory as ``gcorr_total_<tag>_iter<iter>.npz``.
   5. Plot gcorr total and the correction to gcorr total. Save in rundir/plots.
   6. Clear out rundir bandpowers/transfer functions/logs.
   7. Exit.

4. After convergence is reached, copy the gcorr_total file for the last
   iteration from the rundir to the mask directory, labeling it
   ``mask_map_<tag>_gcorr.npz`` for signal or ``mask_map_<tag>_gcorr_null.npz``
   for null.
