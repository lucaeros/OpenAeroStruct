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
    Angles_old,
    Angles,
    measure_angles,
)


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
        Spanwise distribution of the chord scaler.
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
            ref_axis_pos = 0.25  # if no reference axis line is specified : it is the quarter-chord

        mesh = surface["mesh"]
        ny = mesh.shape[1]
        mesh_shape = mesh.shape
        symmetry = surface["symmetry"]

        # This flag determines whether or not changes in z (dihedral) add an
        # additional rotation matrix to modify the twist direction
        self.rotate_x = True

        # 1. Taper
        names = ["taper"]

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

        val = np.ones(ny)
        if "chord_cp" in surface:
            promotes = ["chord"]
            names.append("scale_x")
        else:
            promotes = []

        self.add_subsystem(
            "scale_x",
            ScaleX(val=val, mesh_shape=mesh_shape, ref_axis_pos=ref_axis_pos),
            promotes_inputs=promotes,
        )

        # 3. Sweep

        if "sweep" in surface:
            val = surface["sweep"]
            promotes = ["sweep"]
            names.append("sweep")
        else:
            val = 0.0
            promotes = []

        self.add_subsystem("sweep", Sweep(val=val, mesh_shape=mesh_shape, symmetry=symmetry), promotes_inputs=promotes)

        # 4. Shear X

        val = np.zeros(ny)
        if "xshear_cp" in surface:
            promotes = ["xshear"]
            names.append("shear_x")
        else:
            promotes = []

        self.add_subsystem("shear_x", ShearX(val=val, mesh_shape=mesh_shape), promotes_inputs=promotes)

        # 5. Stretch

        if "span" in surface:
            promotes = ["span"]
            val = surface["span"]
            names.append("stretch")
        else:
            # Compute span. We need .real to make span to avoid OpenMDAO warnings.
            ref_axis = ref_axis_pos * mesh[-1, :, :] + (1 - ref_axis_pos) * mesh[0, :, :]
            span = max(ref_axis[:, 1]).real - min(ref_axis[:, 1]).real
            if symmetry:
                span *= 2.0
            val = span
            promotes = []

        self.add_subsystem(
            "stretch",
            Stretch(val=val, mesh_shape=mesh_shape, symmetry=symmetry, ref_axis_pos=ref_axis_pos),
            promotes_inputs=promotes,
        )

        # 6. Shear Y

        val = np.zeros(ny)
        if "yshear_cp" in surface:
            promotes = ["yshear"]
            names.append("shear_y")
        else:
            promotes = []

        self.add_subsystem("shear_y", ShearY(val=val, mesh_shape=mesh_shape), promotes_inputs=promotes)

        # 7. Dihedral

        if "dihedral" in surface:
            val = surface["dihedral"]
            promotes = ["dihedral"]
            names.append("dihedral")
        else:
            val = 0.0
            promotes = []

        self.add_subsystem(
            "dihedral", Dihedral(val=val, mesh_shape=mesh_shape, symmetry=symmetry), promotes_inputs=promotes
        )

        # 8. Shear Z

        val = np.zeros(ny)
        if "zshear_cp" in surface:
            promotes = ["zshear"]
            names.append("shear_z")
        else:
            promotes = []

        self.add_subsystem("shear_z", ShearZ(val=val, mesh_shape=mesh_shape), promotes_inputs=promotes)
            
        if "angles_cp" in surface:
            promotes = ["angles"]
            val = measure_angles(mesh)
            names.append("angles")
        else:
            promotes = []

        self.add_subsystem(
            "angles",
            Angles_old(mesh_shape=mesh_shape, val = np.zeros(mesh_shape[1]-1), ref_axis_pos=ref_axis_pos),
            promotes_inputs=promotes)

        # 9. Rotate

        val = np.zeros(ny)
        if "twist_cp" in surface:
            promotes = ["twist"]
            names.append("rotate")
        else:
            val = np.zeros(ny)
            promotes = []

        self.add_subsystem(
            "rotate",
            Rotate(val=val, mesh_shape=mesh_shape, symmetry=symmetry, ref_axis_pos=ref_axis_pos),
            promotes_inputs=promotes)
        
        self.promotes(names[-1], outputs=["mesh"])
        if len(names) > 1:
            for j in np.arange(len(names) - 1):
                self.connect(names[j] + ".mesh", names[j + 1] + ".in_mesh")
