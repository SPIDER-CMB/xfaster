import numpy as np
import xfaster as xf
import os

kerns = xf.load_and_parse("outputs_example/95x150/kernels_95x150.npz")
beams = xf.load_and_parse("outputs_example/95x150/beams_95x150.npz")
tf = np.loadtxt("maps_example/transfer_example.txt")
tf.shape
lmax = 500
lk = slice(0, lmax + 1)
bw = beams["beam_windows"]

os.makedirs("transfer_matrix")

for xname in kerns["kern"].keys():
    tag1, tag2 = xname.split(":")
    fname = "transfer_matrix/{}x{}_{{}}_to_{{}}_block.dat".format(tag1, tag2)
    for spec in ["tt", "ee", "bb"]:
        fb2 = tf[lk] * bw[spec][tag1][lk] * bw[spec][tag2][lk]

        if spec == "tt":
            k = kerns["kern"][xname][..., lk]
        else:
            k = kerns["pkern"][xname][..., lk]
            mk = kerns["mkern"][xname][..., lk]

        mat = k * fb2
        su = spec.upper()
        np.savetxt(fname.format(su, su), mat.T)

        if spec != "tt":
            mat = mk * fb2
            su2 = "BB" if spec == "ee" else "EE"
            np.savetxt(fname.format(su, su2), mat.T)
