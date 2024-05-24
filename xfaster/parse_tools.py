""" Module for parsing intermediate and output XFaster files. """
import sys
import os
import numpy as np
from warnings import warn
from collections import OrderedDict

__all__ = [
    "dict_to_arr",
    "arr_to_dict",
    "unique_tags",
    "tag_pairs",
    "dict_to_index",
    "spec_index",
    "spec_mask",
    "dict_to_dmat",
    "dict_to_dsdqb_mat",
    "load_compat",
    "load_pickle_compat",
    "load_and_parse",
    "save",
    "fix_data_roots",
    "corr_index",
    "num_maps",
    "num_corr",
    "expand_qb",
    "bin_spec_simple",
]


def corr_index(idx, n):
    """
    Gets the index of the auto spectrum when getting all
    pairwise combinations of n maps.

    Arguments
    ---------
    idx : int
        The index of the map in the list of maps being looped through.
    n : int
        The number of maps being looped through.

    Returns
    -------
    index : int
        Index of auto spectrum
    """
    return idx * n - idx * (idx - 1) // 2


def num_maps(n):
    """
    Returns how many maps there are if there are n total cross spectra.

    Arguments
    ---------
    n : int
        Number of cross spectra.

    Returns
    -------
    nmaps : int
        Number of maps.
    """
    return int(np.sqrt(8 * n + 1) - 1) // 2


def num_corr(n):
    """
    Returns how many cross spectra there are if there are n total maps.

    Arguments
    ---------
    n : int
        Number of maps.

    Returns
    -------
    nxspec : int
        Number of cross spectra.
    """
    return n * (n + 1) // 2


def unique_tags(tags):
    """
    If map tags are repeated (eg, two 150 maps in different chunk
    subdirectories), return a list modifying them with an index

    Arguments
    ---------
    tags : list of strings
        List of original map tags.

    Returns
    -------
    new_tags : list of strings
        List of map tags where repeated tags are modified to be
        <tag>_<index>, with unique indices.
    """
    if len(np.unique(tags)) == len(tags):
        return tags
    else:
        tags = np.asarray(tags)
        new_tags = []
        indices = {}
        for t in np.unique(tags):
            indices[t] = 0
        for i, m in enumerate(tags):
            if np.count_nonzero(tags == m) > 1:
                # append an index
                new_tags.append("{}_{}".format(m, indices[m]))
                indices[m] += 1
            else:
                new_tags.append(m)
        return new_tags


def tag_pairs(tags, index=False):
    """
    Return an OrderedDict whose keys are pairs of tags in the format "tag1:tag2"
    and whose values are a tuple of the two tags used to construct each key, or
    a tuple of the indices of the two tags in the original tag list, if ``index``
    is True.  If ``index`` is a list, then it should be a list the same length as
    ``tags``, and the tuple is populated by indexing into ``index`` using the two
    indices of the tags in the original tag list.

    Arguments
    ---------
    tags : list of strings
        Map tags from which to construct cross-spectrum keys like "tag1:tag2".
    index : bool
        If True, make values in dictionary the indices of the map tags, rather
        than the tags themselves.

    Returns
    -------
    pairs : OrderedDict
        Dictionary whose keys are pairs of tags in the format "tag1:tag2" and
        whose values are a tuple of the two tags used to construct the key, or
        their indices, if index=True.

    Example
    -------
        >>> tags = ['a', 'b']
        >>> tag_pairs(tags)
        OrderedDict([('a:a', ('a', 'a')), ('a:b', ('a', 'b')), ('b:b', ('b', 'b'))])
        >>> tag_pairs(tags, index=True)
        OrderedDict([('a:a', (0, 0)), ('a:b', (0, 1)), ('b:b', (1, 1))])
        >>> tag_pairs(tags, index=['c', 'd'])
        OrderedDict([('a:a', ('c', 'c')), ('a:b', ('c', 'd')), ('b:b', ('d', 'd'))])
    """
    pairs = OrderedDict()
    for it0, t0 in enumerate(tags):
        for it1, t1 in zip(range(it0, len(tags)), tags[it0:]):
            xname = "{}:{}".format(t0, t1)
            if isinstance(index, list):
                pair = (index[it0], index[it1])
            elif index is True:
                pair = (it0, it1)
            else:
                pair = (t0, t1)
            pairs[xname] = pair
    return pairs


def dict_decode(d):
    """
    Recursively decode key or value bytestrings in a dictionary.
    Useful when loading a bytes-encoded numpy archive file from disk.

    Arguments
    ---------
    d : dict
        Dictionary to decode.

    Returns
    -------
    d2 : dict
        Bytestring-decoded dictionary.
    """
    if not isinstance(d, dict):
        if isinstance(d, bytes):
            return d.decode()
        if isinstance(d, np.ndarray) and d.dtype.char == "S":
            return d.astype(str)
        return d
    d2 = d.__class__()
    for k, v in d.items():
        if isinstance(k, bytes):
            k = k.decode()
        d2[k] = dict_decode(v)
    return d2


def load_compat(*args, **kwargs):
    """
    Load and decode a numpy archive file from disk.

    Backward compatible with python2 data files.

    Arguments
    ---------
    args, kwargs : key/value pairs
        Passed to np.load.

    Returns
    -------
    out : dict
        Dictionary of info from numpy archive file
    """
    if sys.version_info.major > 2:
        kwargs.setdefault("encoding", "latin1")
    if np.__version__ >= "1.16.0":
        kwargs.setdefault("allow_pickle", True)

    out = dict(np.load(*args, **kwargs))

    for k, v in out.items():
        # convert singletons to scalars
        if hasattr(v, "item") and not v.shape:
            v = v.item()

        # handle unicode data
        if sys.version_info.major > 2:
            v = dict_decode(v)

        out[k] = v

    return out


def load_pickle_compat(filename):
    """
    Load a pickle file from the given filename.
    Ensure that the file is open in mode 'rb' (required for python3), and
    that the encoding is set to 'latin1' in python3.

    Arguments
    ---------
    filename : str
        Path to pickled output file to read.

    Returns
    -------
    f : dict
        Unpickled file.
    """
    import pickle

    if hasattr(filename, "read"):
        if filename.mode == "rb":
            try:
                return pickle.load(f, encoding="latin1")
            except TypeError:
                return pickle.load(f)
        warn("Reopening file {} in mode 'rb' for unpickling".format(filename.name))
        filename.close()
        filename = filename.name
    with open(filename, "rb") as f:
        try:
            return pickle.load(f, encoding="latin1")
        except TypeError:
            return pickle.load(f)


def load_and_parse(filename, check_version=True):
    """
    Load a .npz data file from disk and parse all the fields it contains.
    Includes handling of backward compatibility to older file versions
    on disk.

    Returns a dictionary of parsed fields.

    Arguments
    ---------
    filename : str
        Path to numpy data file on disk.
    check_version : bool
        If True, check the data file version and apply any necessary
        updates to the latest version.

    Returns
    -------
    data : dict
        Data dictionary loaded from disk.
    """
    data = load_compat(filename)

    if not check_version:
        return data

    from .xfaster_class import XFaster

    dv = XFaster.data_version
    version = data.get("data_version", -1)
    if version == dv:
        return data

    # backward compatibility
    if version == 1:
        if "raw_root" in data:
            data.pop("raw_root")
            data.pop("raw_files")

        if "data_shape" in data:
            for k in [
                "data_shape",
                "kern_shape",
                "mask_shape",
                "num_corr",
                "num_kern",
                "num_spec",
                "num_spec_mask",
            ]:
                data.pop(k, None)

        # In version 1, these parameters referred to ensembles used for
        # sim_index runs.  In version 2+, these parameters refer to ensembles
        # used for the foreground component of the covariance model
        if "foreground_type" in data:
            data["foreground_type_sim"] = data.pop("foreground_type")
        if "foreground_root" in data:
            data["foreground_root_sim"] = data.pop("foreground_root")
            data["foreground_files_sim"] = data.pop("foreground_files")
        if "foreground_root2" in data:
            data["foreground_root_sim2"] = data.pop("foreground_root2")
            data["foreground_files_sim2"] = data.pop("foreground_files2")
        if "num_foreground" in data:
            data["num_foreground_sim"] = data.pop("num_foreground")

        for k in ["signal_type", "noise_type"]:
            ks = "{}_sim".format(k)
            if k in data and ks in data and data[ks] is None:
                data[ks] = data[k]
                for kk in [
                    "{}_files",
                    "{}_files2",
                    "{}_root",
                    "{}_root2",
                    "num_{}",
                    "num_{}2",
                ]:
                    if kk not in data:
                        continue
                    kk = kk.format(k.split("_")[0])
                    kks = "{}_sim".format(kk)
                    data[kks] = data[kk]

        if "clean_type" in data:
            data["data_type"] = data.pop("clean_type")

        if "planck_root1_hm1" in data:
            ref_root = {
                "ref1a": data.pop("planck_root1_hm1"),
                "ref1b": data.pop("planck_root2_hm1"),
                "ref2a": data.pop("planck_root1_hm2"),
                "ref2b": data.pop("planck_root2_hm2"),
            }
            if all([x is None for x in ref_root.values()]):
                ref_root = None

            ref_files = {
                "ref1a": data.pop("planck_files1_hm1"),
                "ref1b": data.pop("planck_files2_hm1"),
                "ref2a": data.pop("planck_files1_hm2"),
                "ref2b": data.pop("planck_files2_hm2"),
            }
            if all([x is None for x in ref_files.values()]):
                ref_files = None

            data["reference_type"] = None if ref_root is None else "sub"
            data["reference_root"] = ref_root
            data["reference_files"] = ref_files
            data["num_reference"] = data.pop("num_planck", 0)

        if "cls_noise0" in data:
            cls_res = OrderedDict()
            cls_res_null = None

            for k in ["nxn0", "nxn1", "nxs0", "nxs1", "sxn0", "sxn1"]:
                kold = "cls_{}".format(k.replace("nxn", "noise"))
                if data.get(kold, None) is not None:
                    cls_res[k] = data.pop(kold)

                knull = "{}_null".format(kold)
                if data.get(knull, None) is not None:
                    if cls_res_null is None:
                        cls_res_null = OrderedDict()
                    cls_res_null[k] = data.pop(knull)

            data["cls_res"] = cls_res
            data["cls_res_null"] = cls_res_null

        if "cbl" in data:
            cbl = data["cbl"]
            for k in list(cbl):
                if k.startswith("res0") or k.startswith("res1"):
                    knew = "res_{}{}".format(k.split("_")[1], k[3])
                    cbl[knew] = cbl.pop(k)

        if "cls_tnoise_hm1" in data:
            cls_template_noise = OrderedDict()
            cls_template_noise["temp1:temp1"] = data.pop("cls_tnoise_hm1")
            cls_template_noise["temp2:temp2"] = data.pop("cls_tnoise_hm2")
            cls_template_noise["temp1:temp2"] = data.pop("cls_tnoise_hm1xhm2")
            data["cls_template_noise"] = cls_template_noise

        if "template_files" in data:
            if "template_type_sim" not in data:
                for k1, k2 in [
                    ("template_type", "template_type_sim"),
                    ("template_root", "template_root_sim"),
                    ("template_root2", "template_root_sim2"),
                    ("template_files", "template_files_sim"),
                    ("template_files2", "template_files_sim2"),
                    ("num_template", "num_template_sim"),
                ]:
                    if k1 in data:
                        data[k2] = data[k1]

            if "template_noise_type" not in data:
                data["num_template_noise"] = 0
                for k in ["type", "root", "files", "root2", "files2"]:
                    data["template_noise_{}".format(k)] = None

        if "fix_bb_xfer" in data:
            data["fix_bb_transfer"] = data.pop("fix_bb_xfer")

        # update data version in memory
        data["data_version"] = dv

    if version == 2:
        if "reference_root" in data and "reference_type" not in data:
            data["reference_type"] = None if data["reference_root"] is None else "sub"

        # update data version in memory
        data["data_version"] = dv

    if version in [1, 2]:
        if "ref_freq" in data:
            data["freq_ref"] = data.pop("ref_freq")

        if "map_files" in data:
            data["map_names"] = np.asarray(
                [os.path.relpath(f, data["map_root"]) for f in data["map_files"]]
            )

        if "map_files2" in data:
            data["map_names2"] = np.asarray(
                [os.path.relpath(f, data["map_root2"]) for f in data["map_files2"]]
            )

        if "transfer" in data and "qb_transfer" in data:
            xfer = data["transfer"]
            for spec in ["tt", "ee", "bb", "te", "tb", "eb"]:
                if spec in xfer:
                    xfer["cmb_{}".format(spec)] = xfer.pop(spec)

        data["data_version"] = dv

    if version in [1, 2, 3]:
        if "data_root" in data:
            fix_data_roots(data, mode="save")

        data["data_version"] = dv

    version = data.get("data_version", -1)
    if version != dv:
        raise ValueError(
            "Incompatible data file version.  Found {}, expected {}".format(version, dv)
        )

    return data


def save(output_file, **data):
    """
    Save a numpy archive file to disk.

    Arguments
    ---------
    filename : str
        Path to output npz file.
    data : dict
        Dictionary of data to store.
    """
    np.savez_compressed(output_file, **data)


def fix_data_roots(data, mode="save", root=None, root2=None):
    """
    Remove or apply the data root to a set of file paths in an output checkpoint
    file.  Operations are performed in-place on the input dictionary.

    Arguments
    ---------
    data : dict
        Dictionary of file data to update, including lists of map files and file
        roots, as returned by ``XFaster.get_files`` and related functions.
    mode : str
        If ``'save'``, remove the appropriate data root from any file paths, so
        that only relative paths are stored to disk.  If ``'load'``, add the
        appropriate data root to any file paths, so that all paths point to
        existing files within the given data roots.
    root, root2 : str
        The data root (and second data root, for null tests) to be removed or
        applied to the file path items in ``data``.

    Returns
    -------
    data : dict
        Updated data dictionary.
    """

    assert mode in ["load", "save"]

    if root is None and "data_root" in data:
        root = data["data_root"]
    if root2 is None and "data_root2" in data:
        root2 = data["data_root2"]

    def replace_root(k, v):
        if not isinstance(v, str):
            return v
        if root2 is not None and k.endswith("2") and "template" not in k:
            if mode == "load":
                if os.path.isabs(v):
                    return v
                return os.path.join(root2, v)
            elif mode == "save":
                if not v.startswith(root2):
                    return v
                return os.path.relpath(v, root2)
        else:
            if mode == "load":
                if os.path.isabs(v):
                    return v
                return os.path.join(root, v)
            elif mode == "save":
                if not v.startswith(root):
                    return v
                return os.path.relpath(v, root)
        raise RuntimeError("Something has gone horribly wrong, why are you here?")

    for k in list(data):
        if k in ["data_root", "data_root2"]:
            continue
        if "_files" not in k and "_root" not in k:
            continue
        v = data[k]
        if isinstance(v, str):
            data[k] = replace_root(k, v)
        elif isinstance(v, list) and isinstance(v[0], str):
            data[k] = [replace_root(k, vv) for vv in v]
        elif isinstance(v, np.ndarray) and isinstance(v.ravel()[0], str):
            varr = [replace_root(k, vv) for vv in v.ravel()]
            data[k] = np.array(varr).reshape(v.shape)
        elif isinstance(v, dict):
            v1 = list(v.values())[0]
            if isinstance(v1, np.ndarray) and not isinstance(v1.ravel()[0], str):
                continue
            if isinstance(v1, list) and not isinstance(v1[0], str):
                continue
            for kk, vv in v.items():
                vv = np.asarray(vv)
                varr = [replace_root(k, vvv) for vvv in vv.ravel()]
                v[kk] = np.array(varr).reshape(vv.shape)

    return data


def dict_to_arr(d, out=None, flatten=False):
    """
    Transform ordered dict into an array, if all items are same shape

    If not all items are the same shape, eg, for qb, or if flatten=True,
    flatten everything into a vector

    Arguments
    ---------
    d : dict
        Dictionary to transform into an array.
    out : array
        If not None, the starting array onto which to stack the arrays
        contained in dictionary d.
    flatten : bool
        If True, return flattened vector instead of array.

    Returns
    -------
    out : array
        The array containing stacked elements of the arrays contained in
        dictionary ``d``.
    """
    if not isinstance(d, dict):
        return d
    for key, val in d.items():
        if isinstance(val, dict):
            out = dict_to_arr(val, out=out, flatten=flatten)
        else:
            val = np.atleast_1d(val)
            if out is None:
                out = val
            else:
                if val.shape[-1] == out.shape[-1] and not flatten:
                    out = np.vstack([out, val])
                else:
                    out = np.append(out.flatten(), val.flatten())
    return out


def arr_to_dict(arr, ref_dict):
    """
    Transform an array of data into a dictionary keyed by the same keys in
    ref_dict, with data divided into chunks of the same length as in ref_dict.
    Requires that the length of the array is the sum of the lengths of the
    arrays in each entry of ref_dict.  The other dimensions of the input
    array and reference dict can differ.

    Arguments
    ---------
    arr : array
        Input array to be transformed into dictionary.
    ref_dict : dict
        Reference dictionary containing the keys used to construct the output
        dictionary.

    Returns
    -------
    out : dict
        Dictionary of values from arr keyed with keys from ref_dict.
    """
    out = OrderedDict()
    idx = 0
    assert len(arr) == sum([len(v) for v in ref_dict.values()])
    for k, bd in ref_dict.items():
        out[k] = arr[idx : idx + len(bd)]
        idx += len(bd)
    return out


def dict_to_index(d):
    """
    Construct a dictionary of (start, stop) indices that correspond to the
    location of each sub-array when the dict is converted to a single array
    using ``dict_to_arr``.

    Arguments
    ---------
    d : dict
        Input dictionary.

    Returns
    -------
    index : dict
        Dictionary containing location of sub-arrays corresponding to keys.

    Examples
    --------
    To use this function to index into a (nbins, nbins) array, create
    the index dictionary:

    >>> bin_def = OrderedDict((k, np.array([[2, 27], [27, 52]]))
    ...                       for k in ['cmb_tt', 'cmb_ee', 'cmb_bb'])
    >>> bin_index = dict_to_index(bin_def)
    >>> bin_index
    OrderedDict([('cmb_tt', (0, 2)),
                 ('cmb_ee', (2, 4)),
                 ('cmb_bb', (4, 6))])

    To extract the TT bins from the fisher matrix:

    >>> fisher = np.random.randn(12, 12)
    >>> sl_tt = slice(*bin_index['cmb_tt'])
    >>> fisher_tt = fisher[sl_tt, sl_tt]

    To extract all the CMB bins from the fisher matrix:

    >>> sl_cmb = slice(bin_index['cmb_tt'][0], bin_index['cmb_bb'][1])
    >>> fisher_cmb = fisher[sl_cmb, sl_cmb]
    """
    index = OrderedDict()
    idx = 0
    for k, v in d.items():
        index[k] = (idx, idx + len(v))
        idx += len(v)
    return index


def spec_index(spec=None):
    """
    Return the matrix indices of the given spectrum within a 3x3 matrix.  If
    ``spec`` is None, return a dictionary of such indices keyed by spectrum.

    Arguments
    ---------
    spec : str
        Which spectrum to return index for. If None, return dict of all.

    Returns
    -------
    index : dict or list
        Dictionary of indices if spec in None, or list of indices if spec is
        provided.
    """
    inds = OrderedDict(
        [
            ("tt", [0, 0]),
            ("ee", [1, 1]),
            ("bb", [2, 2]),
            ("te", [0, 1]),
            ("eb", [1, 2]),
            ("tb", [0, 2]),
        ]
    )
    if spec is None:
        return inds
    return inds[spec]


def spec_mask(spec=None, nmaps=1):
    """
    Return a mask for extracting spectrum terms from a matrix of shape (3 *
    nmaps, 3 * nmaps).  If ``spec`` is None, returns a dictionary of masks keyed
    by spectrum.

    Arguments
    ---------
    spec : str
        Which spectrum to return mask for. If None, return dict of all masks.
    nmaps : int
        Number of maps used for the cross-spectrum analysis.

    Returns
    -------
    spec_mask : dict or arr
        Dictionary of masks if spec in None, or (3 * nmaps, 3 * nmaps) array
        that is 1 in elements corresponding to spec if spec is provided.
    """
    spec_mask = OrderedDict()

    for s, (i0, i1) in spec_index().items():
        mask = np.zeros((3, 3))
        mask[i0, i1] = mask[i1, i0] = 1
        if nmaps > 1:
            mask = np.tile(mask, (nmaps, nmaps))
        spec_mask[s] = mask

    if spec is None:
        return spec_mask
    return spec_mask[spec]


def dict_to_dmat(dmat_dict, pol=None):
    """
    Take a dmat dictionary and return the right shaped Dmat matrix:
    (Nmaps * 3, Nmaps * 3, lmax + 1) if pol else
    (Nmaps, Nmaps, lmax + 1)

    Arguments
    ---------
    dmat_dict : dict
        Dictionary containing the model covariance terms.

    Returns
    -------
    Dmat : arr
        Dmat total model covariance matrix.
    """
    nmaps = num_maps(len(dmat_dict))

    # get the unique map tags in order from the keys map1:map2
    mtags = [x.split(":")[0] for x in dmat_dict]
    _, uind = np.unique(mtags, return_index=True)
    map_tags = np.asarray(mtags)[sorted(uind)]
    map_pairs = tag_pairs(map_tags, index=True)

    nmaps = len(map_tags)
    pol_dim = 0 if pol is None else (3 if pol else 1)

    Dmat = None
    inds = spec_index()

    for xname, (im0, im1) in map_pairs.items():
        if pol is None:
            pol_dim = 3 if "ee" in dmat_dict[xname] else 1
        for spec, val in dmat_dict[xname].items():
            if Dmat is None:
                shape = (pol_dim * nmaps, pol_dim * nmaps)
                if not np.isscalar(val):
                    shape += val.shape
                Dmat = np.zeros(shape)
            sind = inds[spec]
            xind = im0 * pol_dim + sind[0]
            yind = im1 * pol_dim + sind[1]
            Dmat[xind, yind] = Dmat[yind, xind] = val
            xind = im1 * pol_dim + sind[0]
            yind = im0 * pol_dim + sind[1]
            Dmat[xind, yind] = Dmat[yind, xind] = val

    return Dmat


def dict_to_dsdqb_mat(dsdqb_dict, bin_def):
    """
    Take a dSdqb dictionary and return the right shaped dSdqb matrix:
    (Nmaps * 3, Nmaps * 3, nbins_cmb+nbins_fg+nbins_res, lmax + 1) if pol
    else first two dimensions are Nmaps.

    Arguments
    ---------
    dsdqb_dict : dict
        Dictionary containing the terms for the derivative of the signal
        model, S, w.r.t. the qb parameters.
    bin_def : dict
        Dictionary containing the bin edges for each qb value fit.

    Returns
    -------
    dsdqb_mat : arr
        Signal derivative matrix in the expected shape for matrix multiplication
        in the Fisher iteration.
    """
    # get the unique map tags in order from the keys map1:map2
    mkeys = list(list(dsdqb_dict.values())[0].keys())
    mtags = [x.split(":")[0] for x in mkeys]
    _, uind = np.unique(mtags, return_index=True)
    map_tags = np.asarray(mtags)[sorted(uind)]
    map_pairs = tag_pairs(map_tags, index=True)

    nmaps = len(map_tags)
    pol_dim = 3 if any(["ee" in x.split("_")[1] for x in bin_def]) else 1

    inds = spec_index()
    bin_index = dict_to_index(bin_def)
    nbins = bin_index[list(bin_index)[-1]][-1]

    dsdqb_mat = None
    seen_keys = []

    for key, (start, stop) in bin_index.items():
        bins = slice(start, stop)

        if key == "delta_beta":
            comp = "delta_beta"
            specs = ["tt", "ee", "bb", "te", "eb", "tb"]
            pairs = map_pairs
        else:
            comp, rem = key.split("_", 1)
            if "_" in rem:
                specs, tag = rem.split("_", 1)
                xname = "{0}:{0}".format(tag)
                if xname not in map_pairs:
                    continue
                pairs = {xname: map_pairs[xname]}
                if specs == "eebb":
                    specs = ["ee", "bb"]
                else:
                    specs = [specs]
            else:
                specs = [rem]
                pairs = map_pairs

        if comp not in dsdqb_dict:
            continue

        for xname, (im0, im1) in pairs.items():
            if xname not in dsdqb_dict[comp]:
                continue
            for spec in specs:
                if spec not in dsdqb_dict[comp][xname]:
                    continue
                for spec2, d2 in dsdqb_dict[comp][xname][spec].items():
                    if dsdqb_mat is None:
                        sz = d2.shape[-1]
                        dsdqb_mat = np.zeros(
                            (nmaps * pol_dim, nmaps * pol_dim, nbins, sz)
                        )
                    sind = inds[spec2]
                    ind0 = im0 * pol_dim + sind[0]
                    ind1 = im1 * pol_dim + sind[1]
                    dsdqb_mat[ind0, ind1, bins] = dsdqb_mat[ind1, ind0, bins] = d2
                    ind0 = im1 * pol_dim + sind[0]
                    ind1 = im0 * pol_dim + sind[1]
                    dsdqb_mat[ind0, ind1, bins] = dsdqb_mat[ind1, ind0, bins] = d2
                if key not in seen_keys:
                    seen_keys.append(key)

    # transfer function runs do not include tbeb in the dsdqb matrix
    nbins_seen = max([bin_index[k][-1] for k in seen_keys])
    if nbins_seen != nbins:
        dsdqb_mat = dsdqb_mat[:, :, :nbins_seen, :]

    return dsdqb_mat


def expand_qb(qb, bin_def, lmax=None):
    """
    Expand a qb-type array to an ell-by-ell spectrum using bin_def.

    Arguments
    ---------
    qb : array_like, (nbins,)
        Array of bandpower deviations
    bin_def : array_like, (nbins, 2)
        Array of bin edges for each bin
    lmax : int, optional
        If supplied, limit the output spectrum to this value.
        Otherwise the output spectrum extends to include the last bin.

    Returns
    -------
    cl : array_like, (lmax + 1,)
        Array of expanded bandpowers
    """
    lmax = lmax if lmax is not None else bin_def.max() - 1

    cl = np.zeros(lmax + 1)

    for idx, (left, right) in enumerate(bin_def):
        cl[left:right] = qb[idx]

    return cl


def bin_spec_simple(qb, cls_shape, bin_def, inv_fish=None, lfac=True):
    """
    Compute binned output spectra and covariances by averaging the shape
    spectrum over each bin, and applying the appropriate `qb` bandpower
    amplitude.  NB: this does *not* use the true window functions to compute
    bandpowers, and the results should be treated as an approximation.

    Arguments
    ---------
    qb : dict
        Bandpower amplitudes for each spectrum bin.
    cls_shape : dict
        Shape spectrum
    bin_def : dict
        Bin definition dictionary
    inv_fish : array_like, (nbins, nbins)
        Inverse fisher matrix for computing the bin errors and covariance.  If
        not supplied, these are not computed.
    lfac : bool
        If False, return binned C_l spectrum rather than the default D_l

    Returns
    -------
    cb : dict of arrays
        Binned spectrum
    dcb : dict of arrays
        Binned spectrum error, if `inv_fish` is not None
    ellb : dict of arrays
        Average bin center
    cov : array_like, (nbins, nbins)
        Binned spectrum covariance, if `inv_fish` is not None
    qb2cb : dict
        The conversion factor from `qb` to `cb`, computed by averaging over the
        input shape spectrum.
    """

    lmax = dict_to_arr(bin_def).max()

    qb2cb = OrderedDict()
    ellb = OrderedDict()
    cb = OrderedDict()

    ell = np.arange(lmax + 1)
    fac1 = (2 * ell + 1) / 4.0 / np.pi
    fac2 = ell * (ell + 1) / 2.0 / np.pi
    fac3 = fac1.copy()
    fac3[ell > 0] /= fac2[ell > 0]
    fac = fac1 if lfac else fac3

    ecls_shape = {k: fac * v[: lmax + 1] for k, v in cls_shape.items()}

    bin_index = dict_to_index(bin_def)
    nbins = 0

    for stag, qb1 in qb.items():
        if stag not in ecls_shape:
            continue

        shape = ecls_shape[stag]
        ellb[stag] = np.zeros_like(qb1)
        qb2cb[stag] = np.zeros_like(qb1)

        nbins = max([nbins, bin_index[stag][1]])

        for idx, (left, right) in enumerate(bin_def[stag]):
            il = slice(left, right)
            v = np.sum(shape[il])
            qb2cb[stag][idx] = v / np.sum(fac3[il])
            av = np.abs(shape[il])
            ellb[stag][idx] = np.sum(av * ell[il]) / np.sum(av)

        cb[stag] = qb1 * qb2cb[stag]

    if inv_fish is not None:
        inv_fish = inv_fish[:nbins, :nbins]
        qb2cb_arr = dict_to_arr(qb2cb, flatten=True)
        dcb_arr = np.sqrt(qb2cb_arr * np.abs(np.diag(inv_fish)) * qb2cb_arr)
        dcb = arr_to_dict(dcb_arr, qb2cb)
        cov = np.outer(qb2cb_arr, qb2cb_arr) * inv_fish
    else:
        dcb = None
        cov = None

    return cb, dcb, ellb, cov, qb2cb
