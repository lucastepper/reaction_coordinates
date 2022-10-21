import os
from pathlib import Path
import subprocess
import pytest
import mdtraj as md
import MDAnalysis as mda
import numpy as np
import rcs
from functools import lru_cache
from utils import load_xvg


# # find the test dir in reaction coordinates to create files there
# # ROOTDIR = subprocess.run(['readlink', '-f', rcs.__path__], capture_output=True).stdout
# ROOTDIR = subprocess.run(['readlink', '-f', rcs.__path__[0]], check=True, capture_output=True)
# ROOTDIR = ROOTDIR.stdout.decode('utf8').strip('rcs\n') + 'test/'
# @lru_cache
def get_test_dir() -> Path:
    parent = Path(__file__).parent
    if not str(parent).endswith("test"):
        parent = parent / "test"
    return parent


def compute_hb4_gmx(load_dir: Path, axstart, axend):
    """Compute the hb4 reaction coordinate using gromacs. Creates a temp folder
    in ROOTDIR to save the gromacs helix results and deletes it after finishing."""
    current_dir = os.getcwd()
    temp_dir = get_test_dir() / "temp"
    temp_dir.mkdir(exist_ok=True)
    os.chdir(temp_dir)
    # compute hb4 and other stuff with gmx helix
    subprocess.run(
        [
            "gmx",
            "-quiet",
            "helix",
            "-s",
            get_test_dir() / "ala9/ala9.tpr",
            "-n",
            get_test_dir() / "ala9/ala9.ndx",
            "-ahxstart",
            str(axstart + 1),
            "-ahxend",
            str(axend + 1),
            "-f",
            load_dir,
        ],
        check=True,
    )
    # load to numpy array
    hb4 = load_xvg("hb4.xvg")
    os.chdir(current_dir)
    for file in temp_dir.iterdir():
        file.unlink()
    temp_dir.rmdir()
    return hb4.astype(np.float32)[:, 1]


def compute_hb4_plumed(hb4_obj, xtcfile: Path, mcfile: Path):
    """Compute the hb4 reaction coordinate using plumed driver. Creates a temp folder
    in ROOTDIR to save the plumed results and deletes it after finishing."""
    current_dir = os.getcwd()
    temp_dir = get_test_dir() / "temp"
    if not xtcfile.is_file():
        raise ValueError(f"{xtcfile=} missing.")
    if not mcfile.is_file():
        raise ValueError(f"{mcfile=} missing.")
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    hb4_obj.write_plumed("plumed.dat", rc_file="hb4_plumed.xvg")
    subprocess.run(
        [
            "plumed",
            "driver",
            "--mf_xtc",
            get_test_dir() / xtcfile,
            "--plumed",
            "plumed.dat",
            "--mc",
            get_test_dir() / mcfile,
        ],
        check=True,
    )
    hb4 = load_xvg("hb4_plumed.xvg")
    os.chdir(current_dir)
    for file in temp_dir.iterdir():
        file.unlink()
    temp_dir.rmdir()
    return hb4.astype(np.float32)[:, 1]


@pytest.mark.parametrize(
    "file, axstart, axend",
    [("ala9/ala9", 1, 7)],
)
def test_hb4_against_gmx(file, axstart, axend):
    """Test the computation of the hb4 reaction coordinate against
    the result of gromacs."""
    # u_traj = mda.Universe(str(get_test_dir() / f"{file}.pdb"), str(get_test_dir() / f"{file}.xtc"))
    traj = md.load(get_test_dir() / f"{file}.xtc", top=get_test_dir() / f"{file}.pdb")
    traj.make_molecules_whole()
    hb4 = rcs.HB4(traj, axstart, axend)
    hb4_gmx = compute_hb4_gmx(get_test_dir() / f"{file}.xtc", axstart, axend)
    np.testing.assert_allclose(hb4.compute(), hb4_gmx, atol=1e-6)


@pytest.mark.parametrize(
    "file, axstart, axend",
    [("ala9/ala9", 1, 7)],
)
def test_hb4_against_plumed(file, axstart, axend):
    """Test the computation of the hb4 reaction coordinate against
    the result of plumed driver."""
    traj = md.load(get_test_dir() / f"{file}.xtc", top=get_test_dir() / f"{file}.pdb")
    traj.make_molecules_whole()
    hb4 = rcs.HB4(traj, axstart, axend)
    compute_hb4_plumed(hb4, get_test_dir() / f"{file}.xtc", get_test_dir() / f"{file}_mcfile.txt")
