from copy import deepcopy
import numpy as np
from rcs.reaction_coordinate_base import ReactionCoordinate

class DE2E(ReactionCoordinate):
    """ REaction coordinate that measures the distance between the center of mass of
    the first and last residue, tracking the extension of the protein.
    """
    def __init__(self, traj, top=None):
        """ Constructor Arguments:
            traj (mdtraj.Trajectory or path to one): trajectory or directory that can
            be loaded yielding a mdtraj.Trajectory object.
            top (str): path to topology for the given trajectory. Only needed when traj
            is path to a file without topology information (ie .xtc); default: None
        """
        super().__init__(top=top)
        self.name = 'de2e'
        self.traj = traj
        self._com_start = None
        self._com_end = None
        self.idxs_start = self.traj.topology.select('resid 0')
        self.idxs_end = self.traj.topology.select(f'resid {self.traj.topology.n_residues - 1}')

    def compute_com(self, idxs):
        """ Compute the center of a group of atoms.
        Arguments:
            indexes(indexes): Indexes for the group of atom,
                can be list, range, int, np.ndarray
        """
        pos = self.traj.xyz[:, idxs, :]
        masses = np.array([x.element.mass for x in self.traj.topology.atoms if x.index in idxs])
        return (pos * np.expand_dims(masses, axis=(0, 2))).sum(axis=1) / masses.sum()

    @property
    def com_start(self):
        """ Get the center of mass for the first residue. """
        if self._com_start is None:
            self._com_start = self.compute_com(self.idxs_start)
            return self._com_start
        return self._com_start

    @property
    def com_end(self):
        """ Get the center of mass for the last residue. """
        if self._com_end is None:
            self._com_end = self.compute_com(self.idxs_end)
            return self._com_end
        return self._com_end

    def compute(self):
        """ Computes the distances between the center of mass of
        the first and last residue of the protein"""
        return np.linalg.norm(self.com_start - self.com_end, axis=-1)

    def plot(self, **kwargs):
        """ Plot the end to end distance. Use an quick hack where you write the
        com positions into the positions of the first two hydrogen in the molecule,
        assuming that hydrogens wont be rendered.
        For kwargs documentation, see the documentation for self.get_view
        and self.add_distances.
        """
        hs_to_replace = self.traj.topology.select('mass < 2')[0 : 2]
        traj_copy = deepcopy(self.traj)
        traj_copy.xyz[:, hs_to_replace[0], :] = self.com_start
        traj_copy.xyz[:, hs_to_replace[1], :] = self.com_end
        view = self.get_view(traj_copy, **{k: v for k, v in kwargs.items() if k in self.kwargs_get_view})
        self.add_distances(
            view,
            [[hs_to_replace[0], hs_to_replace[1]]],
            **{k: v for k, v in kwargs.items() if k in self.kwargs_add_distances},
        )
        return view

    def get_lines_plumed(self):
        """ Get the lines needed for the input file to compute
            the de2e RC with plumed programm.
        """
        lines = []
        # remember that plumed starts to count at one
        idxs_start_str = ','.join([str(x + 1) for x in self.idxs_start])
        idxs_end_str = ','.join([str(x + 1) for x in self.idxs_end])
        lines.append(f'com1: COM ATOMS={idxs_start_str} NOPBC \n')
        lines.append(f'com2: COM ATOMS={idxs_end_str} NOPBC \n')
        lines.append(f'{self.name}: DISTANCE ATOMS=com1,com2 NOPBC \n')
        lines.append(' \n')
        return lines
