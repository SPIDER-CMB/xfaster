import sys
import numpy as np
from warnings import warn
from collections import OrderedDict

__all__ = [
    "parse_data",
    "dict_to_arr",
    "arr_to_dict",
    "unique_tags",
    "tag_pairs",
    "dict_to_index",
    "dict_to_dmat",
    "dict_to_dsdqb_mat",
    "load_compat",
    "load_pickle_compat",
    "load_and_parse",
    "corr_index",
    "num_maps",
    "num_corr",
]


def corr_index(idx, n):
    """
    This gets the index of the auto spectrum when getting all
    pairwise combinations of n maps

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
    """
    return int(np.sqrt(8 * n + 1) - 1) // 2


def num_corr(n):
    """
    Returns how many cross spectra there are if there are n total maps.
    """
    return n * (n + 1) // 2


def unique_tags(tags):
    """
    If map tags are repeated (eg, two 150 maps in different chunk
    subdirectories), return a list modifying them with an index
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
    and whose values are a tuple of the two tags used to construct each key,
    or a tuple of the indices of the two tags in the original tag list, if
    `index` is True.

    Example
    -------
        >>> tags = ['a', 'b']
        >>> tag_pairs(tags)
        OrderedDict([('a:a', ('a', 'a')), ('a:b', ('a', 'b')), ('b:b', ('b', 'b'))])
        >>> tag_pairs(tags, index=True)
        OrderedDict([('a:a', (0, 0)), ('a:b', (0, 1)), ('b:b', (1, 1))])
    """
    pairs = OrderedDict()
    for it0, t0 in enumerate(tags):
        for it1, t1 in zip(range(it0, len(tags)), tags[it0:]):
            pairs["{}:{}".format(t0, t1)] = (it0, it1) if index else (t0, t1)
    return pairs


def dict_decode(d):
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
    if sys.version_info.major > 2:
        kwargs.setdefault("encoding", "latin1")
    if np.__version__ >= "1.16.0":
        kwargs.setdefault("allow_pickle", True)

    out = dict(np.load(*args, **kwargs))

    for k, v in out.items():
        # convert singletons to scalars
        if not v.shape:
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
    """
    import pickle
    if hasattr(filename, 'read'):
        if filename.mode == 'rb':
            try:
                return pickle.load(f, encoding='latin1')
            except TypeError:
                return pickle.load(f)
        warn(
            "Reopening file {} in mode 'rb' for unpickling".format(filename.name)
        )
        filename.close()
        filename = filename.name
    with open(filename, 'rb') as f:
        try:
            return pickle.load(f, encoding='latin1')
        except TypeError:
            return pickle.load(f)


def parse_data(data, field, indices=None):
    """
    Look for a field in some data, return as a dictionary with
    descriptive keys.

    Arguments
    ---------
    data : str or dict
        Either the path to an npz file on disk or a loaded npz dict
    field : str
        Which key in data to return as a dictionary.
        Options: bin_def, cls_residual, cbl, qb, fqb, qb_transfer,
                 cb, dcb, dcb_nosampvar, ellb, qb2cb, cls_obs,
                 cls_fg, cls_signal, cls_model, cls_noise, cls_data,
                 cls_shape, wls, w1, w2, w4, fsky, kern, pkern,
                 mkern, xkern, beam_windows, Dmat_obs, Dmat1,
                 dSdqb_mat1
    indices : str
        If given, get the coordinates of this subfield in the original
        bin_def matrix and return a boolean mask where that subfield
        is indexed. Used internally
    """
    if isinstance(data, str):
        data = load_compat(data)

    version = data.get("data_version", -1)
    from .xfaster_class import XFaster
    dv = XFaster.data_version

    # add backward-compatibile parsing here if necessary
    if version != dv:
        raise ValueError(
            "Incompatible data file version.  Found {}, expected {}".format(version, dv)
        )

    # if versions match, then data is already stored in the proper format
    return data[field]


def dict_to_arr(d, out=None, flatten=False):
    """
    Transform ordered dict into an array, if all items are same shape

    If not all items are the same shape, eg, for qb, or if flatten=True,
    flatten everything into a vector
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
    using `dict_to_arr`.

    For example, use this function to index into a (nbins, nbins) array:

        bin_index = dict_to_index(bin_def)

        # extract TT bins from fisher matrix
        sl_tt = slice(*bin_index['cmb_tt'])
        fisher_tt = fisher[sl_tt, sl_tt]

        # extract all CMB bins from fisher matrix
        sl_cmb = slice(bin_index['cmb_tt'][0], bin_index['cmb_tb'][1])
        fisher_cmb = fisher[sl_cmb, sl_cmb]
    """
    index = OrderedDict()
    idx = 0
    for k, v in d.items():
        index[k] = (idx, idx + len(v))
        idx += len(v)
    return index


def dict_to_dmat(dmat_dict):
    """
    Take a dmat dictionary and return the right shaped Dmat matrix:
    (Nmaps * 3, Nmaps * 3, lmax + 1) if pol else
    (Nmaps, Nmaps, lmax + 1)
    """
    nmaps = num_maps(len(dmat_dict))

    # get the unique map tags in order from the keys map1:map2
    mtags = [x.split(":")[0] for x in dmat_dict]
    _, uind = np.unique(mtags, return_index=True)
    map_tags = np.asarray(mtags)[sorted(uind)]
    map_pairs = tag_pairs(map_tags, index=True)

    nmaps = len(map_tags)
    pol_dim = 0

    Dmat = None
    inds = {
        "tt": [0, 0],
        "ee": [1, 1],
        "bb": [2, 2],
        "te": [0, 1],
        "eb": [1, 2],
        "tb": [0, 2],
    }

    for xname, (im0, im1) in map_pairs.items():
        pol_dim = 3 if "ee" in dmat_dict[xname] else 1
        for spec, val in dmat_dict[xname].items():
            if Dmat is None:
                shape = (pol_dim * nmaps, pol_dim * nmaps)
                if not np.isscalar(val):
                    shape += (len(val),)
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

    If gmat is given, the terms in the resulting matrix are multiplied by the
    appriopriate mode density term.
    """
    # get the unique map tags in order from the keys map1:map2
    mtags = [x.split(":")[0] for x in dsdqb_dict["cmb"]]
    _, uind = np.unique(mtags, return_index=True)
    map_tags = np.asarray(mtags)[sorted(uind)]
    map_pairs = tag_pairs(map_tags, index=True)

    nmaps = len(map_tags)
    pol_dim = 3 if "cmb_ee" in bin_def else 1

    inds = {
        "tt": [0, 0],
        "ee": [1, 1],
        "bb": [2, 2],
        "te": [0, 1],
        "eb": [1, 2],
        "tb": [0, 2],
    }

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


def load_and_parse(filename):
    """
    Load a .npz data file from disk and parse all the fields it contains.

    Returns a dictionary of parsed fields.
    """
    data = load_compat(filename)
    ret = dict()
    for k in data:
        ret[k] = parse_data(data, k)
    return ret
