#!/usr/bin/env python
""" A set of classes to store the translational and rotational data from a
hoomd simulation
"""

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
        """ Initialise the values of the TransData object from an array

        Args:
            translations (:class:`numpy.ndarray`): Array containing the
                precomputed translational motion of each molecules
            timesteps (int): The number of timesteps between the initial
                and final configurations
        """
        if isinstance(translations, np.ndarray):
            self.trans = translations
        else:
            self.trans = np.array(translations)
        self.timesteps = timesteps

    def from_json(self, string):
        """Initialise from JSON string

        This is a string encoded in JSON containing at least a list of the
        translational motion of each molecule and the timestep. The following
        is a minimal example ::

            >>> string = "{'timesteps':1, 'translations':[0.1,0.1,0,0.2,0.2]}"
            >>> TransData().from_json(string)

        Todo:
            Deal with error conditions, when the data is incomplete/incorrect

        Args:
            string (string): String in the JSON format containg the data to be
                imported. The json requires both a `translations` field and
                a `timesteps` field.
        """
        dct = json.loads(string)
        self.trans = np.array(dct["translations"])
        self.timesteps = dct["timesteps"]

    def to_json(self, outfile=''):
        """Convert representation to JSON string for writing to a file

        This converts the data to a string which can be easily stored
        in a file for later input and computation. The JSON file structure
        was chosen due to simplicity in both reading and writing as well
        as the descriptive nature.

        Note:
            While each string is the the JSON format the entire file is
            not in the JSON format. This is due to the expected large size
            of the files, making it not feasible to store the entire file in
            memory at the same time. Since it is simple enough to deal with
            a single frame at a time, each frame will be on a separate line
            which is in the JSON format and the file can be dealt with line
            by line.

        Args:
            outfile (string): Filename to direct data to
        """
        output = open(outfile, 'a')
        l_trans = self.trans.tolist()
        output.write(json.dumps({"__TransData__":True,
                                 "timesteps":self.timesteps,
                                 "translations":l_trans,
                                }, output, separators=(',', ':')))
        output.write('\n')
        output.close()



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
        self.bodies = 0

    def from_arrays(self, trans, rot, timesteps, bodies=0):
        """Initialise TransRotData from precomputed :class:`numpy.ndarray`

        Both the translational and rotational arrays have to have the same
        ordering of molecules i.e. the data in `trans[i]` corresponds to the
        same molecue as `rot[i]`

        Todo:
            Deal with error condiditions where there are not the same
            number of molecules etc.

        Args:
            trans (:class:`numpy.ndarray`): Array of all translations
            rot (:class:`numpy.ndarray`): Array of all rotations
            timesteps (int): Number of timesteps between initial and final
                configurations.
            bodies (int): The number of rigid bodies in the system
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
        if bodies == 0:
            self.bodies = len(self.trans)
        else:
            self.bodies = bodies

    def from_json(self, string):
        """Initialise from JSON string

        This is a string encoded in JSON containing at least a list of the
        translational motion of each molecule, the rotational motion of each
        molecule and the timestep. The following is a minimal example ::

            >>> string = "{'timesteps':1, 'translations':[0.1,0.1,0,0.2,0.2],
            'rotations':[0.1,0.1,0,0.2,0.2]}"
            >>> TransRotData().from_json(string)

        Todo:
            Deal with error conditions, when the data is incomplete/incorrect

        Args:
            string (string): String in the JSON format containg the data to be
                imported. The json requires a `translations` field, a
                `rotations` field and a `timesteps` field.
        """
        dct = json.loads(string)
        self.trans = np.array(dct["translations"])
        self.rot = np.array(dct["rotations"])
        self.timesteps = dct["timesteps"]
        if dct.get("bodies", 0):
            self.bodies = bodies
        else:
            self.bodies = len(self.trans)

    def to_json(self, outfile=''):
        """Convert representation to JSON string for writing to a file

        This converts the data to a string which can be easily stored
        in a file for later input and computation. The JSON file structure
        was chosen due to simplicity in both reading and writing as well
        as the descriptive nature.

        Note:
            While each string is the the JSON format the entire file is
            not in the JSON format. This is due to the expected large size
            of the files, making it not feasible to store the entire file in
            memory at the same time. Since it is simple enough to deal with
            a single frame at a time, each frame will be on a separate line
            which is in the JSON format and the file can be dealt with line
            by line.

        Args:
            outfile (string): Filename to direct data to
        """
        if outfile == '':
            output = sys.stdout
        else:
            output = open(outfile, 'a')
        l_trans = self.trans.tolist()
        l_rot = np.ndarray.tolist(self.rot)
        output.write(json.dumps({"__TransRotData__":True,
                                 "timesteps":self.timesteps,
                                 "bodies": self.bodies,
                                 "translations":l_trans,
                                 "rotations":l_rot
                                }, output, separators=(',', ':')))
        output.write('\n')
        output.close()

