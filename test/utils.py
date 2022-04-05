import os
from pathlib import Path
import subprocess
import shlex
import shutil
import toml
import numpy as np


def get_test_dir() -> Path:
    """ Assume that we are starting test either by running pyling cmd
    in project base dir or from 'test' dir. Find out which one it is
    and return relative path.
    """
    if os.path.isdir("test"):
        return Path("test")
    return Path("")


def get_temp_dir():
    """ Get a temporary directory to save test data in. """
    # assume we are in linux system, pytest fixture did not work when importing to different file
    tmp_dir = Path("/tmp/tmp_rcs_test_data")
    if not tmp_dir.is_dir():
        tmp_dir.mkdir()
    return tmp_dir



def load_xvg(file, max_rows=1000):
    """ Load a plumed .xvg file output. """
    output = None
    skiped_rows = 0
    while output is None:
        try:
            output = np.loadtxt(file, skiprows=skiped_rows)
        except ValueError:
            skiped_rows += 1
        if skiped_rows > max_rows:
            raise ValueError(f'Could not load {file}, the first {max_rows} lines were misformated')
    return output


def purge_temp_dir():
    """ Get a temporary directory to save test data in. """
    # assume we are in linux system, pytest fixture did not work when importing to different file
    tmp_dir = Path("/tmp/tmp_rcs_test_data")
    shutil.rmtree(tmp_dir)


def run_subprocess(cmd_str: str):
    """ Run a gmx or plumed command in a subprocess"""
    # prepare env
    env = {}
    config = toml.load(get_test_dir() / "gmx_config.toml")
    for source_key in ["gmx_source", "plumed_source"]:
        cmd = shlex.split(f"env -i bash -c 'source {config[source_key]} && env'")
        output = subprocess.run(cmd, check=True, capture_output=True).stdout.decode()
        for line in output.split("\n"):
            if "=" in line:
                key, item = line.split("=")
                if key in env:
                    env[key] += f":{item}"
                else:
                    env[key] = item
    # run subprocess
    subprocess.run(shlex.split(cmd_str), check=True, capture_output=True, env=env)


def compute_rc_plumed(rc_obj, xtc_file, mc_file):
    """ Compute an rc using plumed in a subprocess.
    Arguments:
        rc_obj: instance of rcs.BaseReactionCoordinate
        xtc_file: str, path to example trajectory in 'test' dir
        mc_file: str, path to plumed mass charge file in 'test' dir
    """
    # save input file
    save_name = get_temp_dir() / 'plumed_result.xvg'
    rc_obj.write_plumed(get_temp_dir() / 'plumed.dat', rc_file=save_name)
    # run cmd
    cmd = f"plumed driver --mf_xtc {get_test_dir() / xtc_file}"
    cmd += f" --plumed {get_temp_dir() / 'plumed.dat'}"
    if mc_file:
        cmd += f" --mc {get_test_dir() / mc_file}"
    run_subprocess(cmd)
    # load results
    output = load_xvg(save_name)
    purge_temp_dir()
    return output.astype(np.float32)[:, 1]
