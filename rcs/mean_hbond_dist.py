from copy import deepcopy
import numpy as np
import mdtraj as md
from rcs.reaction_coordinate_base import ReactionCoordinate


class MeanContactDistance(ReactionCoordinate):
    """ Computes a mean distance between contacting residues
    based on the distance between their center of masses
    or on the two hydrogen bonds that can form. The hbond 
    length is measured by the backbone distances N1->O2 and N2->O1
    """
    def __init__(self, traj, mode, resnames=(), chains=(), name='mean_contact', top=None):
        """ Constructor Arguments:
            traj (mdtraj.Trajectory): trajectory
            resnames (list of lists): list of lists, containing pairs of contacts
            chains (list of list or str): list containing chain1 and chain 2 for 
                contacts, if all the same, or list of lists.
            name (str): name for the reaction coordinate; default: mean_hbond_dist
        """
        super().__init__(top=top)
        self.name = name
        self.traj = traj
        self.mode = mode
        self._contacts = []
        self._com_contacts = []
        if len(chains) > 1 and isinstance(chains[0], int) and isinstance(chains[1], int):
            chains = [(chains[0], chains[1]) for __ in range(len(resnames))]
        for chain, resname in zip(chains, resnames):
            self.add_contact(resname1=resname[0], resname2=resname[1], chainid1=chain[0], chainid2=chain[1])
     
            
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode in ['hbonds', 'h-bonds', 'h_bonds', 'h-bond', 'hbond', 'h_bond']:
            self._mode = 'hbonds'
        elif mode in ['com', 'center-of-mass', 'centre-of-mass', 'center_of_mass', 'centre_of_mass']:
            self._mode = 'com'
        else:
            raise ValueError('mode needs to be either "com" or "hbonds".')
              
    @property
    def com_contacts(self):
        """ Get the center of mass for all contacting residues """
        if not self._com_contacts:
            for contact1, contact2 in self._contacts:
                self._com_contacts.append(
                    (self.compute_com(contact1), self.compute_com(contact2))
                )
        return self._com_contacts

            
    def _get_bb_idx_by_element(self, idxs, element):
        """ Given a list of indexs, get the corresponding atom index for a backbone atom
        of the given element from self.traj.topology. Assumes only one match exists.
        """
        for idx in idxs:
            atom = self.traj.topology.atom(idx)
            if atom.is_backbone and atom.element.name == element:
                return idx

    def add_contact(self, resid1=None, resid2=None, resname1=None, resname2=None, chainid1=None, chainid2=None):
        """ Indexes 0-based. """
        # check that residues 1 and 2 have an identifier
        idxs_res1 = self._identify_residue(resid1, resname1, chainid1)
        idxs_res2 = self._identify_residue(resid2, resname2, chainid2)
        # get N O indexes, if mode hbond
        top = self.traj.topology
        if self.mode == 'hbonds':
            self._contacts.extend([
                (
                    self._get_bb_idx_by_element(idxs_res1, 'nitrogen'), 
                    self._get_bb_idx_by_element(idxs_res2, 'oxygen'),
                ),
                (
                    self._get_bb_idx_by_element(idxs_res2, 'nitrogen'), 
                    self._get_bb_idx_by_element(idxs_res1, 'oxygen'),
                ),
            ])
        elif self.mode == 'com':
            self._contacts.append((idxs_res1, idxs_res2))

    def _identify_residue(self, resid, resname, chainid):
        """ Check that given the description for the one of the residues
        in a contact only identifies a single residue. 
        """
        if resid is None and resname is None:
            raise ValueError(f'Please identify the {name} residue involved in contact either by name or by id.')
        select_str = ''
        if resname is not None:
            select_str += f'resname {resname} and '
        if resid is not None:
            select_str += f'residue {resid} and '
        if chainid is not None:
            select_str += f'chainid {chainid} and'
        select_str =  select_str.strip(' and')
        top = self.traj.topology
        idxs_res = top.select(select_str)
        # check that all belong to the same residue
        if len(idxs_res) == 0:
            raise ValueError(f'Given the selection information {select_str}, no residue was found.')
        idx_first = top.atom(idxs_res[0]).residue.index
        residues_identified = list(set([top.atom(i).residue.index for i in idxs_res]))
        if len(residues_identified) > 1:
            raise ValueError(
                f'Given the selection information {select_str}, more than one'
                f'residue was identified, these were resid 0-based: {residues_identified}'
            )
        return idxs_res

    def compute(self):
        if self.mode == 'hbonds':
            return md.compute_distances(self.traj, np.array(self._contacts)).mean(-1)
        elif self.mode == 'com':
            rc = np.zeros(len(self.traj))
            for com1, com2 in self.com_contacts:
                rc += np.linalg.norm(com1 - com2, axis=-1)
            return rc / len(self.com_contacts)
        
    def plot(self, view=None, **kwargs):
        """ Plot the contacting residues, either the hbonds or from com1 to com2.
        Arguments:
            view (nglview.NGLWidget): scene to render the bonds into; default None
                if None, creates a new scene
            Accepts the kwargs for self.get_view and self.add_distances. 
        """
        if not view:
            view = self.get_view(self.traj, **{k: v for k, v in kwargs.items() if k in ['opacity']})
        if self.mode == 'hbonds':
            for __ in self._contacts:
                self.add_distances(
                    view,
                    self._contacts,
                    **{k: v for k, v in kwargs.items() if k in ['color', 'label_color']},
                )
        elif self.mode == 'com':
            hs_to_replace = self.traj.topology.select('mass < 2')[ : 2 * len(self.com_contacts)]
            traj_copy = deepcopy(self.traj)
            for i, (com1, com2) in enumerate(self.com_contacts):
                traj_copy.xyz[:, hs_to_replace[2 * i], :] = com1
                traj_copy.xyz[:, hs_to_replace[2 * i + 1], :] = com2
            view = self.get_view(traj_copy, **{k: v for k, v in kwargs.items() if k in self.kwargs_get_view})
            for i in range(len(self.com_contacts)):
                self.add_distances(
                    view,
                    [[hs_to_replace[2 * i], hs_to_replace[2 * i + 1]]],
                    **{k: v for k, v in kwargs.items() if k in self.kwargs_add_distances},
                )
        return view

    def get_lines_plumed(self):
        """ Get the lines needed for the input file to compute
            the RC with plumed programm.
        """
        lines = []
        if self.mode == 'hbonds':
            for i, (idx1, idx2) in enumerate(self._contacts):
                lines.append(f'cont{i + 1}: DISTANCE ATOMS={int(idx1) + 1},{int(idx2) + 1} NOPBC \n')
            lines.append(' \n')
            lines.append(f'{self.name}: CUSTOM ...  \n')
            lines.append('ARG={}  \n'.format(','.join([f'cont{i + 1}' for i in range(len(self._contacts))])))
            lines.append('VAR={}  \n'.format(','.join([f'a{i}' for i in range(len(self._contacts))])))
            lines.append(
                'FUNC={'
                + '({}) / '.format(' + '.join([f'a{i}' for i in range(len(self._contacts))]))
                + '%i} \n'%len(self._contacts)
            )
            lines.append('PERIODIC=NO \n')
            lines.append('... \n')
        elif self.mode == 'com':
            for i, (idxs_com1, idxs_com2) in enumerate(self._contacts):
                idxs_com1 = ','.join([str(x + 1) for x in idxs_com1])
                idxs_com2 = ','.join([str(x + 1) for x in idxs_com2])
                lines.append(f'com{2 * i + 1}: COM ATOMS={idxs_com1} NOPBC \n')
                lines.append(f'com{2 * i + 2}: COM ATOMS={idxs_com2} NOPBC \n')
                lines.append(f'd{i + 1}: DISTANCE ATOMS=com{2 * i + 1},com{2 * i + 2} NOPBC \n')
            lines.append(' \n')
            lines.append(f'{self.name}: CUSTOM ...  \n')
            lines.append('ARG={}  \n'.format(','.join([f'd{i + 1}' for i in range(len(self._contacts))])))
            lines.append('VAR={}  \n'.format(','.join([f'a{i}' for i in range(len(self._contacts))])))
            lines.append(
                'FUNC={'
                + '({}) / '.format(' + '.join([f'a{i}' for i in range(len(self._contacts))]))
                + '%i} \n'%len(self._contacts)
            )
            lines.append('PERIODIC=NO \n')
            lines.append('... \n')
        lines.append(' \n')
        return lines

MeanHbondDistance = MeanContactDistance