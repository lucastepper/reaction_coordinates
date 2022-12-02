import os
from abc import abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np
import MDAnalysis as mda
import nglview as nv


class ReactionCoordinate():
    """ Base class for reaction coordinate.
    Children need to define name and traj attributs. """

    name: str

    def __init__(self, top: Optional[str]=None):
        self.kwargs_get_view = ['frames', 'opacity', 'selection', 'superpose_to_frame']
        self.kwargs_add_distances = ['color', 'label_color']
        self._top = top
        self._universe = None

    @property
    def universe(self) -> mda.Universe:
        """ Get the universe obj if available. """
        if self._universe is None:
            raise ValueError('universeectory not set up yet')
        else:
            return self._universe

    @property
    def natoms(self) -> int:
        """ Get the universe obj if available. """
        return len(self.universe.atoms)

    @universe.setter
    def universe(self, universe: mda.Universe|str|Path):
        """ Set the universe obj either from a MDAnalysis.Universe obj or by
        loading with the constructor, possibly using self._top as topology """
        if isinstance(universe, (str, Path)):
            if not self._top:
                self._universe = mda.Universe(str(universe))
            else:
                self._universe = mda.Universe(self._top, str(universe))
        elif isinstance(universe, mda.Universe):
            self._universe = universe
        else:
            raise ValueError(
                'universe needs to be path to a loadable file'
                'or a MDAnalysis.Universe object. '
            )

    @abstractmethod
    def compute(self):
        raise NotImplementedError

    @abstractmethod
    def get_lines_plumed(self):
        pass

    def compute_com(self, idxs, stride=1):
        """ Compute the center of a group of atoms.
        Arguments:
            indexes(indexes): Indexes for the group of atom,
                can be list, range, int, np.ndarray
        """
        com = np.zeros(u)
        masses = np.array([x.element.mass for x in self.traj.topology.atoms if x.index in idxs])
        return (pos * np.expand_dims(masses, axis=(0, 2))).sum(axis=1) / masses.sum()

    def idxs_to_str(self, idxs):
        """ Convert a set of indexes to a string containing the indexes comma separated.
        If more than 2 indexes are consecutive, convert them into a range of the from
        start-end (inclusve on both sides). This format is accepted by plumed for
        atom selection. Assumes indexes are zero bases and increments them by one.
        Arguments:
            idxs (iterable of ints) Indexes. """

        # items are either ints for single idxs or strings for ranges
        items = []
        idxs = list(idxs)
        # increment idxs for mdtraj -> plumed conversion
        idxs = sorted([int(x) + 1 for x in idxs])
        while len(idxs) > 0:
            n_consecutive = 0
            for i in range(len(idxs) - 1):
                if idxs[i] + 1 != idxs[i + 1]:
                    break
                n_consecutive += 1
            if n_consecutive > 1:
                items.append((idxs[0], idxs[n_consecutive]))
                for __ in range(n_consecutive + 1):
                    idxs.pop(0)
            else:
                items.append(idxs.pop(0))
        # convert ranges to idx_start-idxs_end, ints to strings, join by comma
        return ','.join([str(x) if isinstance(x, int) else f'{x[0]}-{x[1]}' for x in items])

    def write_plumed(self, file_name, rc_file=None, append=False, overwrite=True, stride=1):
        """ Write the lines for plumed into a file.
        Arguments:
            file_name (str): path for plumed input file to write.
            rc_file (str): path for plumed to write the result; default None
                if none, no output will be generated.
            append (bool): append to an existing plumed input file; default False.
            overwrite (bool): overwrite a plumed input file if exists; default True.
            stride (int): spacing between frames for plumed to write the output.
        """
        if not overwrite and os.path.isfile(file_name):
            raise FileExistsError('File exists and overwrite is False.')
        if append and not os.path.isfile(file_name):
            raise FileNotFoundError('File needs to exist in order to append.')
        lines = self.get_lines_plumed()
        if rc_file:
            lines.append(f'PRINT ARG={self.name} STRIDE={int(stride)} FILE={rc_file} \n')
            lines.append(' \n')
        with open(file_name, 'a' if append else 'w') as fh:
            for line in lines:
                fh.write(line)

    def add_distance(self, view, p1, p2, color=(1, 0, 1), thickness=0.1, n_dash=11, factor=10):
        """ Render a bond as a dahsed cylinder in a nglview.View object This is slow and only works
        for a single frame (as it is not moving with the frames for multiple).
        Arguments:
            view (nglview.View) scene in which to render the bond
            p1 (np.ndarray): bond start
            p2 (np.ndarray): bond end
            color (tuple, len=3): RBG code for color; default (1, 0, 1)
            thickness (float): thickness of the cylinder rendered; default 0.1
            n_dash (int): number of dashes of the cylinder; default 11
            factor (float): conversion factor for the points. nglview uses Angstrom
            gromacs nm; default 10 (nm -> A)
        """
        p1 = factor * np.copy(p1.flatten())
        p2 = factor * np.copy(p2.flatten())
        v_12 = p2.flatten() - p1.flatten()
        for i in range(0, n_dash, 2):
            view.shape.add_cylinder(
                (p1 + i / n_dash * v_12).tolist(), (p1 + (i + 1) / n_dash * v_12).tolist(), color, thickness
            )

    def add_distances(self, view, atom_pairs, color='magenta', label_color='magenta'):
        """ Renders distances between atoms in a nglview.View object. The corresponding
        distance is indicated in the View object in Angstrom. Colors can be select
        as any string that nglview accepts (I could not find a docu for that.)
        Arguments:
            view (nglview.View) scene in which to render the bond
            atom_pairs (list of lists): atom pairs to connect via a bond
            color (str): color of the bond
            label_color (str): color of the label
        """
        view.add_distance(
            atom_pair=[[float(x[0]), float(x[1])] for x in atom_pairs],
            color=color,
            label_color=label_color,
        )

    def get_view(self, traj, frames=None, opacity=0.8, selection='backbone', superpose_to_frame=1):
        """ Get a nglview.View scene and plot the chosen frames as a cartoon
        and the backbone as licorice. The coloring scheme used is based
        on the residue index of the chain segment and starts with red
        at the N-terminus going to blue at the C-terminus.
        Arguments:
            traj (mdtraj.Trajectory): traj to render
            opacity (float): opacity for rendering the backbone atoms
            frames (indexes): indexes to select only some frames of traj; default None
                Can be int, list, np.ndarray or range. if None, all frames are shown
            selection (str): selection for the licorice that is rendered, for reference,
            see: https://mdtraj.org/1.9.4/atom_selection.html
            superpose_to_frame (int): superpose the plotted trajectory to the this frame; default 1
        """
        traj = traj.superpose(traj[min(0, superpose_to_frame - 1)])
        if not frames:
            view = nv.show_mdtraj(traj)
        else:
            view = nv.show_mdtraj(traj[frames])
        selection_idxs = self.traj.topology.select(selection)
        view.clear()
        view.add_representation('cartoon', color='residueindex', selection='protein')
        view.add_representation('licorice', color='residueindex', selection=selection_idxs, opacity=opacity)
        return view
