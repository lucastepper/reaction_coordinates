import os
import subprocess
import pytest
import mdtraj as md
import numpy as np
import rcs


# find the test dir in reaction coordinates to create files there
ROOTDIR = subprocess.run(['readlink', '-f', rcs.__path__[0]], check=True, capture_output=True)
ROOTDIR = ROOTDIR.stdout.decode('utf8').strip('rcs\n') + 'test/'


def compute_rg_gmx(xtc_file, tpr_file, ndx_file):
    """ Compute the rg reaction coordinate using gromacs. Creates a temp folder
    in ROOTDIR to save the gromacs gyrate results and deletes it after finishing. """
    current_dir = os.getcwd()
    temp_dir = ROOTDIR + 'temp'
    assert np.all([os.path.isfile(ROOTDIR + x) for x in [xtc_file, tpr_file, ndx_file]])
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    # compute hb4 and other stuff with gmx helix
    subprocess.run([
        'gmx', 'gyrate',
        '-f', ROOTDIR + xtc_file,
        '-s', ROOTDIR + tpr_file,
        '-n', ROOTDIR + ndx_file,
        '-o', 'rg.xvg'],
        check=True,
        capture_output=True,
    )
    # load to numpy array
    rg = rcs.load_xvg('rg.xvg')
    os.chdir(current_dir)
    for file in os.listdir(temp_dir):
        os.remove(temp_dir + '/' + file)
    os.rmdir(temp_dir)
    return rg.astype(np.float32)[:, 1]


def compute_rg_plumed(rg_obj, xtc_file, mc_file):
    """ Compute the rg reaction coordinate using plumed driver. Creates a temp folder
    in ROOTDIR to save the plumed results and deletes it after finishing. """
    current_dir = os.getcwd()
    temp_dir = ROOTDIR + 'temp'
    assert np.all([os.path.isfile(ROOTDIR + xtc_file) for x in [xtc_file, mc_file]])
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    rg_obj.write_plumed('plumed.dat', rc_file='rg_plumed.xvg')
    subprocess.run([
         'plumed', 'driver',
         '--mf_xtc', ROOTDIR + xtc_file,
         '--plumed', 'plumed.dat',
         '--mc', ROOTDIR + mc_file],
        check=True,
    )
    rg = rcs.load_xvg('rg_plumed.xvg')
    os.chdir(current_dir)
    for file in os.listdir(temp_dir):
        os.remove(temp_dir + '/' + file)
    os.rmdir(temp_dir)
    return rg.astype(np.float32)[:, 1]


@pytest.mark.parametrize(
    'xtc_file, tpr_file, ndx_file, selection',
    [
        ('ala9/ala9.xtc', 'ala9/ala9.tpr', 'ala9/ala9_heavy.ndx', 'mass > 2')
    ],
)
def test_rg_against_gmx(xtc_file, tpr_file, ndx_file, selection):
    """ Test the computation of the rg reaction coordinate against
    the result of gromacs. """
    traj = md.load(ROOTDIR + xtc_file, top=ROOTDIR + xtc_file.strip('.xtc') + '.pdb')
    traj.make_molecules_whole()
    rg = rcs.RG(traj, selection=selection)
    rg_gmx = compute_rg_gmx(xtc_file, tpr_file, ndx_file)
    np.testing.assert_allclose(rg.compute(), rg_gmx, atol=5e-5)


@pytest.mark.parametrize(
    'xtc_file, selection',
    [
        ('ala9/ala9.xtc', 'mass > 2')
    ],
)
def test_rg_against_plumed(xtc_file, selection):
    """ Test the computation of the rg reaction coordinate against
    the result of plumed driver. """
    traj = md.load(ROOTDIR + xtc_file, top=ROOTDIR + xtc_file.strip('.xtc') + '.pdb')
    traj.make_molecules_whole()
    rg = rcs.RG(traj, selection=selection)
    compute_rg_plumed(rg, xtc_file, xtc_file.strip('.xtc') + '_mcfile.txt')







