import os
import subprocess
import pytest
import mdtraj as md
import numpy as np
import utils
import rcs


def compute_com_ref(top, idxs):
    """Compute the center of a group of atoms.
    Arguments:
        top (str): path to topology file
        indexes(indexes): Indexes for the group of atom,
            can be list, range, int, np.ndarray
    """
    traj = md.load(utils.get_test_dir() / top, top=utils.get_test_dir() / top)
    if idxs is None:
        idxs = np.arange(traj.topology.n_atoms)
    pos = traj.xyz[:, idxs, :]
    masses = np.array([x.element.mass for x in traj.topology.atoms if x.index in idxs])
    return (pos * np.expand_dims(masses, axis=(0, 2))).sum(axis=1) / masses.sum()


@pytest.mark.parametrize(
    "top, stride, idxs",
    [
        ("ala9/ala9.pdb", 1, None),
        ("ala9/ala9.pdb", 12, None),
        ("ala9/ala9.pdb", 1, np.arange(20)),
        ("ala9/ala9.pdb", 12, np.arange(20)),
    ],
)
def test_com(top, stride, idxs):
    """Test the center of mass calculation"""
    # we build a reaction coordinate object to get the the, com method of ReactionCoordinateBase
    hb4 = rcs.HB4(utils.get_test_dir() / top, axstart=0, axend=9)
    com = hb4.compute_com(idxs=idxs, stride=stride)
    com_ref = compute_com_ref(top, idxs=idxs)[::stride]
    np.testing.assert_allclose(com, com_ref, atol=1e-5)


if __name__ == "__main__":
    test_com("ala9/ala9.pdb", 1, None)
