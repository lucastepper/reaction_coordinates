import os
import subprocess
import pytest
import mdtraj as md
import numpy as np
import rcs


# find the test dir in reaction coordinates to create files there
# ROOTDIR = subprocess.run(['readlink', '-f', rcs.__path__], capture_output=True).stdout
ROOTDIR = subprocess.run(['readlink', '-f', rcs.__path__[0]], check=True, capture_output=True)
ROOTDIR = ROOTDIR.stdout.decode('utf8').strip('rcs\n') + 'test/'


def compute_hb4_gmx(load_dir):
    """ Compute the hb4 reaction coordinate using gromacs. Creates a temp folder
    in ROOTDIR to save the gromacs helix results and deletes it after finishing. """
    current_dir = os.getcwd()
    temp_dir = ROOTDIR + 'temp'
    assert os.path.isfile(load_dir)
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    # compute hb4 and other stuff with gmx helix
    subprocess.run([
        'gmx', '-quiet', 'helix',
        '-s', ROOTDIR + 'ala9/ala9.tpr',
        '-n', ROOTDIR + 'ala9/ala9.ndx',
        '-ahxstart', '2',
        '-ahxend', '8',
        '-f', load_dir],
        check=True,
    )
    # load to numpy array
    hb4 = rcs.load_xvg('hb4.xvg')
    os.chdir(current_dir)
    for file in os.listdir(temp_dir):
        os.remove(temp_dir + '/' + file)
    os.rmdir(temp_dir)
    return hb4.astype(np.float32)[:, 1]


def compute_hb4_plumed(hb4_obj, xtcfile, mcfile):
    """ Compute the hb4 reaction coordinate using plumed driver. Creates a temp folder
    in ROOTDIR to save the plumed results and deletes it after finishing. """
    current_dir = os.getcwd()
    temp_dir = ROOTDIR + 'temp'
    assert os.path.isfile(xtcfile)
    assert os.path.isfile(mcfile)
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    os.chdir(temp_dir)
    hb4_obj.write_plumed('plumed.dat', rc_file='hb4_plumed.xvg')
    subprocess.run([
         'plumed', 'driver',
         '--mf_xtc', ROOTDIR + xtcfile,
         '--plumed', 'plumed.dat',
         '--mc', ROOTDIR + mcfile],
        check=True,
    )
    hb4 = rcs.load_xvg('hb4_plumed.xvg')
    os.chdir(current_dir)
    for file in os.listdir(temp_dir):
        os.remove(temp_dir + '/' + file)
    os.rmdir(temp_dir)
    return hb4.astype(np.float32)[:, 1]


@pytest.mark.parametrize(
    'file, axstart, axend',
    [
        ('ala9/ala9', 2, 8)
    ],
)
def test_hb4_against_gmx(file, axstart, axend):
    """ Test the computation of the hb4 reaction coordinate against
    the result of gromacs. """
    traj = md.load(f'{ROOTDIR}{file}.xtc', top=f'{ROOTDIR}{file}.pdb')
    traj.make_molecules_whole()
    hb4 = rcs.HB4(traj, axstart, axend)
    hb4_gmx = compute_hb4_gmx(f'{ROOTDIR}{file}.xtc')
    np.testing.assert_allclose(hb4.compute(), hb4_gmx, atol=1e-6)


@pytest.mark.parametrize(
    'file, axstart, axend',
    [
        ('ala9/ala9', 2, 8)
    ],
)
def test_hb4_against_plumed(file, axstart, axend):
    """ Test the computation of the hb4 reaction coordinate against
    the result of plumed driver. """
    traj = md.load(f'{ROOTDIR}{file}.xtc', top=f'{ROOTDIR}{file}.pdb')
    traj.make_molecules_whole()
    hb4 = rcs.HB4(traj, axstart, axend)
    compute_hb4_plumed(hb4, file + '.xtc', file + '_mcfile.txt')







