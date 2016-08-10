#!/usr/bin/env python
""" A set of classes used for computing the dynamic properties of a Hoomd MD
simulation"""

from __future__ import print_function
import sys
import json
import numpy as np

class TransData(object):
    """Class to deal with the translational data for computation

    The dynamics quantities we are concerned with calculating are all computed
    from the displacements of atoms. Which means we can store all the
    information we need for computations in the translation of each atom and
    the time difference

    """
    def __init__(self):
        super(TransData, self).__init__()
        self.trans = np.array([])
        self.timesteps = 0

    def from_trans_array(self, translations, timesteps):
        """ Initialise the values of the TransData object from a list

        Args:
            translations (:numpy:`array`): Array containing the precomputed
                translational motion of each molecules
            timesteps (int): The number of timesteps between the initial
                and final configurations
        """
        if isinstance(translations, np.ndarray):
            self.trans = translations
        else:
            self.trans = np.array(translations)
        self.timesteps = timesteps

    def from_json(self, string):
        """Initialise from JSON string"""
        dct = json.loads(string)
        self.trans = np.array(dct["translations"])
        self.timesteps = dct["timesteps"]

    def to_json(self, outfile=''):
        """Convert representation to JSON for writing to a file
        """
        if outfile:
            output = sys.stdout
        else:
            output = open(outfile, 'w')
        json.dump({"__TransRotData__":True,
                   "timesteps":self.timesteps,
                   "translations":np.ndarray.tolist(self.trans)
                  }, output, separators=(',', ':'))



class TransRotData(TransData):
    """Class to hold the translational and rotational data for computation

    The dyanmics quantities that we are interesrted in are all computed from
    the translations and rotations of the individual molecules. This means
    that we can compute any of the values of interest from just knowing the
    translation and rotation of each molecule for a given time difference.

    """
    def __init__(self):
        super(TransRotData, self).__init__()
        self.rot = np.array([])

    def from_arrays(self, trans, rot, timesteps):
        """Initialise TransRotData from precomputed arrays

        Both the translational and rotational arrays have to have the same
        ordering of molecules i.e. the data in `trans[i]` corresponds to the
        same molecue as `rot[i]`

        Args:
            trans (array): Array of all translations
            rot (array): Array of all rotations
            timesteps (int): Number of timesteps between initial and final
                configurations.

        """
        if isinstance(trans, np.ndarray):
            self.trans = trans
        else:
            self.trans = np.array(trans)
        assert isinstance(trans, np.ndarray)
        if isinstance(rot, np.ndarray):
            self.rot = rot
        else:
            self.rot = np.array(rot)
        assert isinstance(rot, np.ndarray)
        self.timesteps = timesteps

    def from_json(self, string):
        """Initialise from JSON string"""
        dct = json.loads(string)
        self.trans = np.array(dct["translations"])
        self.rot = np.array(dct["rotations"])
        self.timesteps = dct["timesteps"]

    def to_json(self, outfile=''):
        """Convert representation to JSON for writing to a file
        """
        if outfile == '':
            output = sys.stdout
        else:
            output = open(outfile, 'a')
        l_trans = self.trans.tolist()
        l_rot = np.ndarray.tolist(self.rot)
        output.write(json.dumps({"__TransRotData__":True,
                                 "timesteps":self.timesteps,
                                 "translations":l_trans,
                                 "rotations":l_rot
                                }, output, separators=(',', ':')))
        output.write('\n')
        output.close()

