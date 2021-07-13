
# pylint: disable=no-member
import os
from abc import abstractmethod
import numpy as np
import nglview as nv


class ReactionCoordinate():
    """ Base class for reaction coordinate.
    Children need to define name and traj attributs. """

    @abstractmethod
    def compute(self):
        raise NotImplementedError

    @abstractmethod
    def get_lines_plumed(self):
        pass

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
        """ Render a bond as a dahsed cylinder in a nglview.View object.
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

    def get_view(self, frames=(0, 1), opacity=0.8):
        """ Get a nglview.View scene and plot the chosen frames as a cartoon
        and the backbone as licorice. The coloring scheme used is based
        on the residue index of the chain segment and starts with red
        at the N-terminus going to blue at the C-terminus.
        Arguments:
            frames (tuple len=2): frames to plot, set is inclusive left, exclusive right
                default (0, 1); which is the first frame
            opacity (float): opacity for rendering the backbone atoms
        """
        view = nv.show_mdtraj(self.traj[frames[0] : frames[1]])
        view.clear()
        view.add_representation('cartoon', color='residueindex', selection='protein')
        view.add_representation('licorice', color='residueindex', selection='backbone', opacity=opacity)
        return view
