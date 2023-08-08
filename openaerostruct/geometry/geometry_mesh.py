""" Group that manipulates geometry mesh based on high-level design parameters. """

import numpy as np

import openmdao.api as om

from openaerostruct.geometry.geometry_mesh_transformations import (
    Taper,
    ScaleX,
    Sweep,
    ShearX,
    Stretch,
    ShearY,
    Dihedral,
    ShearZ,
    Rotate,
    Angles,
    measure_angles,
    measure_twist,
)
import warnings


class GeometryMesh(om.Group):
    """
    OpenMDAO group that performs mesh manipulation functions. It reads in
    the initial mesh from the surface dictionary and outputs the altered
    mesh based on the geometric design variables.

    Depending on the design variables selected or the supplied geometry information,
    only some of the follow parameters will actually be given to this component.
    If parameters are not active (they do not deform the mesh), then
    they will not be given to this component.

    Parameters
    ----------
    sweep : float
        Shearing sweep angle in degrees.
    dihedral : float
        Dihedral angle in degrees.
    twist[ny] : numpy array
        1-D array of rotation angles for each wing slice in degrees.
    chord_dist[ny] : numpy array
        Chord length for each panel edge.
    taper : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point at the tip.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Modified mesh based on the initial mesh in the surface dictionary and
        the geometric design variables.
    """

    def initialize(self):
        self.options.declare("surface", types=dict)

    def setup(self):
        surface = self.options["surface"]

        if "ref_axis_pos" in surface:
            ref_axis_pos = surface["ref_axis_pos"]
        else:
            ref_axis_pos = 0.25  # if no reference axis line is specified : it is the quarter of chord

        mesh = surface["mesh"]
        ny = mesh.shape[1]
        mesh_shape = mesh.shape
        symmetry = surface["symmetry"]

        # This flag determines whether or not changes in z (dihedral) add an
        # additional rotation matrix to modify the twist direction
        self.rotate_x = True
        names = ["taper"]
        # 1. Taper

        if "taper" in surface:
            val = surface["taper"]
            promotes = ["taper"]
        else:
            val = 1.0
            promotes = []

        self.add_subsystem(
            "taper", Taper(val=val, mesh=mesh, symmetry=symmetry, ref_axis_pos=ref_axis_pos), promotes_inputs=promotes
        )

        # 2. Scale X
        names.append("scale_x")
        val = np.ones(ny)
        if "chord_cp" in surface:
            promotes = ["chord"]
        else:
            promotes = []

        self.add_subsystem(
            "scale_x",
            ScaleX(val=val, mesh_shape=mesh_shape, ref_axis_pos=ref_axis_pos),
            promotes_inputs=promotes,
        )

        # 3. Sweep
        names.append("sweep")
        if "sweep" in surface:
            val = surface["sweep"]
            promotes = ["sweep"]
        else:
            val = 0.0
            promotes = []

        self.add_subsystem("sweep", Sweep(val=val, mesh_shape=mesh_shape, symmetry=symmetry), promotes_inputs=promotes)

        # 4. Shear X
        names.append("shear_x")
        val = np.zeros(ny)
        if "xshear_cp" in surface:
            promotes = ["xshear"]
        else:
            promotes = []

        self.add_subsystem("shear_x", ShearX(val=val, mesh_shape=mesh_shape), promotes_inputs=promotes)

        # 5. Stretch
        if "span" in surface:
            promotes = ["span"]
            val = surface["span"]
            names.append("stretch")
            self.add_subsystem(
                "stretch",
                Stretch(val=val, mesh_shape=mesh_shape, symmetry=symmetry, ref_axis_pos=ref_axis_pos),
                promotes_inputs=promotes,
            )

        # 6. Shear Y
        names.append("shear_y")
        val = np.zeros(ny)
        if "yshear_cp" in surface:
            promotes = ["yshear"]
        else:
            promotes = []

        self.add_subsystem("shear_y", ShearY(val=val, mesh_shape=mesh_shape), promotes_inputs=promotes)

        # 7. Dihedral
        names.append("dihedral")
        if "dihedral" in surface:
            val = surface["dihedral"]
            promotes = ["dihedral"]
        else:
            val = 0.0
            promotes = []

        self.add_subsystem(
            "dihedral", Dihedral(val=val, mesh_shape=mesh_shape, symmetry=symmetry), promotes_inputs=promotes
        )

        if "angles_cp" in surface:
            promotes = ["angles"]
            val = np.zeros(ny - 1)
            names.append("angles")
            self.add_subsystem(
                "angles", Angles(val=val, mesh_shape=mesh.shape, ref_axis_pos=ref_axis_pos), promotes_inputs=promotes
            )
        else:
            promotes = []
            te = mesh[-1]
            le = mesh[0]
            ref_axis = ref_axis_pos * te + (1 - ref_axis_pos) * le
            dz_qc = ref_axis[:-1, 2] - ref_axis[1:, 2]
            dy_qc = ref_axis[:-1, 1] - ref_axis[1:, 1]
            val = np.arctan(dz_qc / dy_qc) * 180 / np.pi

        # 8. Shear Z

        val = np.zeros(ny)
        names.append("shear_z")
        if "zshear_cp" in surface:
            promotes = ["zshear"]
        else:
            promotes = []

        self.add_subsystem(
            "shear_z", ShearZ(val=val, mesh_shape=mesh_shape), promotes_inputs=promotes, promotes_outputs=["mesh"]
        )

        # 9. Rotate

        if "twist_cp" in surface:
            promotes = ["twist"]
            self.add_subsystem(
                "rotate",
                Rotate(val=val, mesh_shape=mesh_shape, symmetry=symmetry, ref_axis_pos=ref_axis_pos),
                promotes_inputs=promotes,
                promotes_outputs=["mesh"],
            )
        else:
            promotes = []

        for j in np.arange(len(names) - 1):
            self.connect(names[j] + ".mesh", names[j + 1] + ".in_mesh")
