""" A set of components that manipulate geometry mesh
    based on high-level design parameters. """

from re import S
import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd, jacrev, jit
from jax.config import config

config.update("jax_enable_x64", True)
import openmdao.api as om


def measure_angles(mesh, ref_axis_pos=0.25):
    te = mesh[-1]
    le = mesh[0]
    quarter_chord = ref_axis_pos * te + (1 - ref_axis_pos) * le
    dz_qc = quarter_chord[:-1, 2] - quarter_chord[1:, 2]
    dy_qc = quarter_chord[:-1, 1] - quarter_chord[1:, 1]
    theta_x = np.arctan(dz_qc / dy_qc) * 180 / np.pi
    # Prepend with 0 so that root is not rotated
    return theta_x


def measure_twist(mesh):
    xaxis = np.array([1, 0, 0])
    te = mesh[-1]
    le = mesh[0]
    quarter_chord = 0.25 * te + 0.75 * le
    dz_qc = quarter_chord[:-1, 2] - quarter_chord[1:, 2]
    dy_qc = quarter_chord[:-1, 1] - quarter_chord[1:, 1]
    theta_x = np.arctan(dz_qc / dy_qc)
    theta_x = np.append(theta_x, 0.0)
    phi_y = np.zeros(len(theta_x))
    for k in range(len(theta_x)):
        normal = np.array([0, np.cos(theta_x[k]), np.sin(theta_x[k])])
        LEq = mesh[0, k, :] - quarter_chord[k, :]
        LEq = LEq / np.linalg.norm(LEq)
        phi_y[k] = np.arcsin(np.dot(np.cross(LEq, xaxis), normal)) * 180 / np.pi
    return phi_y


def apply_angles(mesh, angles, ref_axis_pos=0.25):
    mesh_shape = mesh.shape
    ref_axis = ref_axis_pos * mesh[-1, :, :] + (1 - ref_axis_pos) * mesh[0, :, :]
    vect = ref_axis[:-1, :] - ref_axis[1:, :]
    new_ref_axis = ref_axis.copy()
    new_mesh = mesh.copy()
    for j in reversed(range(mesh_shape[1] - 1)):
        norms = np.sqrt(vect[j, 1] ** 2 + vect[j, 2] ** 2)
        new_ref_axis[j, :] = new_ref_axis[j + 1, :] + np.array(
            [
                vect[j, 0],
                norms * np.cos(angles[j] * np.pi / 180),
                norms * np.sin(angles[j] * np.pi / 180),
            ]
        )
        new_mesh[:, j, 1:] = new_ref_axis[j, 1:]
    return new_mesh


def apply_angles_old(mesh, angles):
    mesh_shape = mesh.shape
    xsection = jnp.array([1, 0, 0])
    te = mesh[-1]
    le = mesh[0]
    s0 = mesh_shape[0]
    s1 = mesh_shape[1]
    quarter_chord = 0.25 * te + 0.75 * le
    vect = quarter_chord[:-1, :] - quarter_chord[1:, :]
    new_quarters = jnp.zeros(quarter_chord.shape)
    new_quarters = new_quarters.at[s1 - 1, :].set(quarter_chord[s1 - 1, :])
    new_mesh = jnp.zeros(mesh_shape)
    new_mesh = new_mesh.at[:, -1, :].set(mesh[:, -1, :])
    for j in reversed(range(s1 - 1)):
        norms = jnp.sqrt(vect.at[j, 1].get() ** 2 + vect.at[j, 2].get() ** 2)
        new_quarters = new_quarters.at[j, :].set(
            new_quarters[j + 1, :]
            + jnp.array(
                [
                    vect[j, 0],
                    norms * jnp.cos(angles[j] * np.pi / 180),
                    norms * jnp.sin(angles[j] * np.pi / 180),
                ]
            )
        )
        newvect = new_quarters[j, :] - new_quarters[j + 1, :]
        ysection = vect[j] / jnp.linalg.norm(vect[j])
        zsection = jnp.cross(ysection, xsection)
        newysection = newvect / jnp.linalg.norm(newvect)
        newzsection = jnp.cross(newysection, xsection)
        for i in range(s0):
            points = mesh[i, j, :] - quarter_chord[j, :]
            z = jnp.dot(points, zsection)
            new_mesh = new_mesh.at[i, j, :].set(new_quarters[j, :] + points[0] * xsection + z * newzsection)
    return new_mesh


class AnglesOld(om.ExplicitComponent):
    def initialize(self):
        """
        Declare options.
        """
        # self.options.declare("mesh")
        self.options.declare("mesh_shape", desc="mesh")
        self.options.declare("val", desc="Initial value for angles dihedral")
        self.options.declare(
            "ref_axis_pos",
            default=0.25,
            desc="Fraction of the chord that is use as the reference axis line",
        )

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]
        self.add_input("in_mesh", val=np.zeros(mesh_shape), units="m")
        self.add_input("angles", val=val, units="deg")
        self.add_output("mesh", val=np.zeros(mesh_shape), units="m")
        self.declare_partials(of="mesh", wrt="in_mesh")
        self.declare_partials(of="mesh", wrt="angles")
        self.func = jit(apply_angles)
        self.J1 = jit(jacfwd(apply_angles, 0))
        self.J2 = jit(jacfwd(apply_angles, 1))

    def compute(self, inputs, outputs):
        in_mesh = jnp.array(inputs["in_mesh"])
        angles = jnp.array(inputs["angles"])
        mesh_shape = jnp.array(self.options["mesh_shape"])
        outputs["mesh"] = self.func(in_mesh, angles)

    def compute_partials(self, inputs, J):
        in_mesh = jnp.array(inputs["in_mesh"])
        angles = jnp.array(inputs["angles"])
        mesh_shape = jnp.array(self.options["mesh_shape"])
        p = np.prod(mesh_shape)
        J1 = self.J1(in_mesh, angles)
        J2 = self.J2(in_mesh, angles)
        J["mesh", "in_mesh"] = J1.reshape((p, p))
        J["mesh", "angles"] = J2.reshape((p, mesh_shape[1] - 1))


class DiffTwist(om.ExplicitComponent):
    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("N", desc="number of value for twist")

    def setup(self):
        N = self.options["N"]
        self.N = N
        self.add_input("dtwist_cp", val=np.zeros(N))
        self.add_output("twist_cp", val=np.zeros(N))
        self.declare_partials(of="twist_cp", wrt="dtwist_cp")

    def compute(self, inputs, outputs):
        outputs["twist_cp"][-1] = inputs["dtwist_cp"][-1]
        for k in range(1, self.N):
            outputs["twist_cp"][-1 - k] = inputs["dtwist_cp"][-k] + inputs["dtwist_cp"][-k - 1]

    def compute_partials(self, inputs, J):
        J["twist_cp", "dtwist_cp"][-1, -1] = 1
        for k in range(1, self.N):
            J["twist_cp", "dtwist_cp"][-1 - k, -k] = 1
            J["twist_cp", "dtwist_cp"][-1 - k, -k - 1] = 1


class Angles(om.ExplicitComponent):
    def initialize(self):
        """
        Declare options.
        """
        # self.options.declare("mesh")
        self.options.declare("mesh_shape", desc="mesh")
        self.options.declare("val", desc="Initial value for angles dihedral")
        self.options.declare(
            "ref_axis_pos",
            default=0.25,
            desc="Fraction of the chord that is use as the reference axis line",
        )

    def setup(self):
        self.mesh_shape = self.options["mesh_shape"]
        self.val = self.options["val"]
        self.ref_axis_pos = self.options["ref_axis_pos"]
        self.add_input("in_mesh", val=np.zeros(self.mesh_shape), units="m")
        self.add_input("angles", val=self.val, units="deg")
        self.add_output("mesh", val=np.zeros(self.mesh_shape), units="m")
        self.declare_partials(of="mesh", wrt="in_mesh", method="cs")
        self.declare_partials(of="mesh", wrt="angles", method="cs")

    def compute(self, inputs, outputs):
        mesh = inputs["in_mesh"]
        angles = inputs["angles"]
        ref_axis = self.ref_axis_pos * mesh[-1, :, :] + (1 - self.ref_axis_pos) * mesh[0, :, :]
        vect = ref_axis[:-1, :] - ref_axis[1:, :]
        new_ref_axis = ref_axis.copy()
        new_mesh = mesh.copy()
        mesh_shape = mesh.shape
        xsection = np.zeros(3)
        xsection[0] = 1
        for j in reversed(range(mesh_shape[1] - 1)):
            norms = np.sqrt(vect[j, 1] ** 2 + vect[j, 2] ** 2)
            new_ref_axis[j, :] = new_ref_axis[j + 1, :] + np.array(
                [
                    vect[j, 0],
                    norms * np.cos(angles[j] * np.pi / 180),
                    norms * np.sin(angles[j] * np.pi / 180),
                ]
            )
            new_vect = new_ref_axis[j, :] - new_ref_axis[j + 1, :]
            ysection = vect[j] / np.linalg.norm(vect[j])
            zsection = np.cross(ysection, xsection)
            newysection = new_vect / jnp.linalg.norm(new_vect)
            newzsection = np.cross(newysection, xsection)
            for i in range(mesh_shape[0]):
                points = mesh[i, j, :] - ref_axis[j, :]
                z = np.dot(points, zsection)
                new_mesh[i, j, :] = new_ref_axis[j, :] + points[0] * xsection + z * newzsection
        outputs["mesh"] = new_mesh


class Angles_old(om.ExplicitComponent):
    def initialize(self):
        """
        Declare options.
        """
        # self.options.declare("mesh")
        self.options.declare("mesh_shape", desc="mesh")
        self.options.declare("val", desc="Initial value for angles dihedral")
        self.options.declare(
            "ref_axis_pos",
            default=0.25,
            desc="Fraction of the chord that is use as the reference axis line",
        )

    def setup(self):
        self.mesh_shape = self.options["mesh_shape"]
        self.val = self.options["val"]
        self.ref_axis_pos = self.options["ref_axis_pos"]
        self.add_input("in_mesh", val=np.zeros(self.mesh_shape), units="m")
        self.add_input("angles", val=self.val, units="deg")
        self.add_output("mesh", val=np.zeros(self.mesh_shape), units="m")
        self.declare_partials(of="mesh", wrt="in_mesh", method="cs")
        self.declare_partials(of="mesh", wrt="angles", method="cs")

    def compute(self, inputs, outputs):
        mesh = inputs["in_mesh"]
        angles = inputs["angles"]
        ref_axis = self.ref_axis_pos * mesh[-1, :, :] + (1 - self.ref_axis_pos) * mesh[0, :, :]
        vect = ref_axis[:-1, :] - ref_axis[1:, :]
        new_ref_axis = ref_axis.copy()
        new_mesh = mesh.copy()
        for j in reversed(range(self.mesh_shape[1] - 1)):
            norms = np.sqrt(vect[j, 1] ** 2 + vect[j, 2] ** 2)
            new_ref_axis[j, :] = new_ref_axis[j + 1, :] + np.array(
                [
                    vect[j, 0],
                    norms * np.cos(angles[j] * np.pi / 180),
                    norms * np.sin(angles[j] * np.pi / 180),
                ]
            )
            new_mesh[:, j, 1:] = new_ref_axis[j, 1:]
        outputs["mesh"] = new_mesh


class Taper(om.ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by altering the spanwise chord linearly to produce
    a tapered wing. Note that we apply taper around the reference axis line which is quarter of chord by default.

    Parameters
    ----------
    taper : float
        Taper ratio for the wing; 1 is untapered, 0 goes to a point at the tip.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the tapered aerodynamic surface.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("val", desc="Initial value for the taper ratio.")
        self.options.declare("mesh", desc="Nodal mesh defining the initial aerodynamic surface.")
        self.options.declare(
            "symmetry", default=False, desc="Flag set to true if surface is reflected about y=0 plane."
        )
        self.options.declare(
            "ref_axis_pos",
            default=0.25,
            desc="Fraction of the chord that is use as the reference axis line",
        )

    def setup(self):
        mesh = self.options["mesh"]
        val = self.options["val"]
        self.ref_axis_pos = self.options["ref_axis_pos"]

        self.add_input("taper", val=val)

        self.add_output("mesh", val=mesh, units="m")

        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        mesh = self.options["mesh"]
        symmetry = self.options["symmetry"]
        taper_ratio = inputs["taper"][0]

        # Get mesh parameters and the quarter-chord
        le = mesh[0]
        te = mesh[-1]
        num_x, num_y, _ = mesh.shape
        ref_axis = self.ref_axis_pos * te + (1 - self.ref_axis_pos) * le
        x = ref_axis[:, 1]
        span = x[-1] - x[0]

        # If symmetric, solve for the correct taper ratio, which is a linear
        # interpolation problem
        if symmetry:
            xp = np.array([-span, 0.0])
            fp = np.array([taper_ratio, 1.0])

        # Otherwise, we set up an interpolation problem for the entire wing, which
        # consists of two linear segments
        else:
            xp = np.array([-span / 2, 0.0, span / 2])
            fp = np.array([taper_ratio, 1.0, taper_ratio])

        taper = np.interp(x.real, xp.real, fp.real)

        # Modify the mesh based on the taper amount computed per spanwise section
        outputs["mesh"] = np.einsum("ijk,j->ijk", mesh - ref_axis, taper) + ref_axis

    def compute_partials(self, inputs, partials):
        mesh = self.options["mesh"]
        symmetry = self.options["symmetry"]
        taper_ratio = inputs["taper"][0]

        # Get mesh parameters and the quarter-chord
        le = mesh[0]
        te = mesh[-1]
        num_x, num_y, _ = mesh.shape
        ref_axis = self.ref_axis_pos * te + (1 - self.ref_axis_pos) * le
        x = ref_axis[:, 1]
        span = x[-1] - x[0]

        # If symmetric, solve for the correct taper ratio, which is a linear
        # interpolation problem
        if symmetry:
            xp = np.array([-span, 0.0])
            fp = np.array([taper_ratio, 1.0])

        # Otherwise, we set up an interpolation problem for the entire wing, which
        # consists of two linear segments
        else:
            xp = np.array([-span / 2, 0.0, span / 2])
            fp = np.array([taper_ratio, 1.0, taper_ratio])

        taper = np.interp(x, xp, fp)

        if taper_ratio == 1.0:
            dtaper = np.zeros(taper.shape)
        else:
            dtaper = (1.0 - taper) / (1.0 - taper_ratio)

        partials["mesh", "taper"] = np.einsum("ijk, j->ijk", mesh - ref_axis, dtaper)


class ScaleX(om.ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by modifying the chords along the span of the
    wing by scaling only the x-coord.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    chord[ny] : numpy array
        Chord length for each panel edge.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("val", desc="Initial value for chord lengths")
        self.options.declare("mesh_shape", desc="Tuple containing mesh shape (nx, ny).")
        self.options.declare(
            "ref_axis_pos",
            default=0.25,
            desc="Fraction of the chord that is use as the reference axis line",
        )

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]
        self.ref_axis_pos = self.options["ref_axis_pos"]
        self.add_input("chord", units="m", val=val)
        self.add_input("in_mesh", shape=mesh_shape, units="m")

        self.add_output("mesh", shape=mesh_shape, units="m")

        nx, ny, _ = mesh_shape
        nn = nx * ny * 3

        rows = np.arange(nn)
        col = np.tile(np.zeros(3), ny) + np.repeat(np.arange(ny), 3)
        cols = np.tile(col, nx)

        self.declare_partials("mesh", "chord", rows=rows, cols=cols)

        p_rows = np.arange(nn)
        te_rows = np.arange(((nx - 1) * ny * 3))
        le_rows = te_rows + ny * 3
        le_cols = np.tile(np.arange(3 * ny), nx - 1)
        te_cols = le_cols + ny * 3 * (nx - 1)
        rows = np.concatenate([p_rows, te_rows, le_rows])
        cols = np.concatenate([p_rows, te_cols, le_cols])

        self.declare_partials("mesh", "in_mesh", rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        mesh = inputs["in_mesh"]
        chord_dist = inputs["chord"]

        te = mesh[-1]
        le = mesh[0]
        ref_axis = self.ref_axis_pos * te + (1 - self.ref_axis_pos) * le

        outputs["mesh"] = np.einsum("ijk,j->ijk", mesh - ref_axis, chord_dist) + ref_axis

    def compute_partials(self, inputs, partials):
        mesh = inputs["in_mesh"]
        chord_dist = inputs["chord"]

        te = mesh[-1]
        le = mesh[0]
        ref_axis = self.ref_axis_pos * te + (1 - self.ref_axis_pos) * le

        partials["mesh", "chord"] = (mesh - ref_axis).flatten()

        nx, ny, _ = mesh.shape
        nn = nx * ny * 3
        d_mesh = np.einsum("i,ij->ij", chord_dist, np.ones((ny, 3))).flatten()
        partials["mesh", "in_mesh"][:nn] = np.tile(d_mesh, nx)

        d_qc = (np.einsum("ij,i->ij", np.ones((ny, 3)), 1.0 - chord_dist)).flatten()
        nnq = (nx - 1) * ny * 3
        partials["mesh", "in_mesh"][nn : nn + nnq] = np.tile(self.ref_axis_pos * d_qc, nx - 1)
        partials["mesh", "in_mesh"][nn + nnq :] = np.tile((1 - self.ref_axis_pos) * d_qc, nx - 1)

        nnq = ny * 3
        partials["mesh", "in_mesh"][nn - nnq : nn] += self.ref_axis_pos * d_qc
        partials["mesh", "in_mesh"][:nnq] += (1 - self.ref_axis_pos) * d_qc


class Sweep(om.ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh applying shearing sweep. Positive sweeps back.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    sweep : float
        Shearing sweep angle in degrees.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the swept aerodynamic surface.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("val", desc="Initial value for x shear.")
        self.options.declare("mesh_shape", desc="Tuple containing mesh shape (nx, ny).")
        self.options.declare(
            "symmetry", default=False, desc="Flag set to true if surface is reflected about y=0 plane."
        )

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]

        self.add_input("sweep", val=val, units="deg")
        self.add_input("in_mesh", shape=mesh_shape, units="m")

        self.add_output("mesh", shape=mesh_shape, units="m")

        nx, ny, _ = mesh_shape
        nn = nx * ny
        rows = 3 * np.arange(nn)
        cols = np.zeros(nn)

        self.declare_partials("mesh", "sweep", rows=rows, cols=cols)

        nn = nx * ny * 3
        n_rows = np.arange(nn)

        if self.options["symmetry"]:
            y_cp = ny * 3 - 2
            te_cols = np.tile(y_cp, nx * (ny - 1))
            te_rows = np.tile(3 * np.arange(ny - 1), nx) + np.repeat(3 * ny * np.arange(nx), ny - 1)
            se_cols = np.tile(3 * np.arange(ny - 1) + 1, nx)
        else:
            y_cp = 3 * (ny + 1) // 2 - 2
            n_sym = (ny - 1) // 2

            te_row = np.tile(3 * np.arange(n_sym), 2) + np.repeat([0, 3 * (n_sym + 1)], n_sym)
            te_rows = np.tile(te_row, nx) + np.repeat(3 * ny * np.arange(nx), ny - 1)

            te_col = np.tile(y_cp, n_sym)
            se_col1 = 3 * np.arange(n_sym) + 1
            se_col2 = 3 * np.arange(n_sym) + 4 + 3 * n_sym

            # neat trick: swap columns on reflected side so we can assign in just two operations
            te_cols = np.tile(np.concatenate([te_col, se_col2]), nx)
            se_cols = np.tile(np.concatenate([se_col1, te_col]), nx)

        rows = np.concatenate(([n_rows, te_rows, te_rows]))
        cols = np.concatenate(([n_rows, te_cols, se_cols]))

        self.declare_partials("mesh", "in_mesh", rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        symmetry = self.options["symmetry"]
        sweep_angle = inputs["sweep"][0]
        mesh = inputs["in_mesh"]

        # Get the mesh parameters and desired sweep angle
        nx, ny, _ = mesh.shape
        le = mesh[0]
        p180 = np.pi / 180
        tan_theta = np.tan(p180 * sweep_angle)

        # If symmetric, simply vary the x-coord based on the distance from the
        # center of the wing
        if symmetry:
            y0 = le[-1, 1]
            dx = -(le[:, 1] - y0) * tan_theta

        # Else, vary the x-coord on either side of the wing
        else:
            ny2 = (ny - 1) // 2
            y0 = le[ny2, 1]

            dx_right = (le[ny2:, 1] - y0) * tan_theta
            dx_left = -(le[:ny2, 1] - y0) * tan_theta
            dx = np.hstack((dx_left, dx_right))

        # dx added spanwise.
        outputs["mesh"][:] = mesh
        outputs["mesh"][:, :, 0] += dx

    def compute_partials(self, inputs, partials):
        symmetry = self.options["symmetry"]
        sweep_angle = inputs["sweep"][0]
        mesh = inputs["in_mesh"]

        # Get the mesh parameters and desired sweep angle
        nx, ny, _ = mesh.shape
        le = mesh[0]
        p180 = np.pi / 180
        tan_theta = np.tan(p180 * sweep_angle)
        dtan_dtheta = p180 / np.cos(p180 * sweep_angle) ** 2

        # If symmetric, simply vary the x-coord based on the distance from the
        # center of the wing
        if symmetry:
            y0 = le[-1, 1]

            dx_dtheta = -(le[:, 1] - y0)

        # Else, vary the x-coord on either side of the wing
        else:
            ny2 = (ny - 1) // 2
            y0 = le[ny2, 1]

            dx_dtheta_right = le[ny2:, 1] - y0
            dx_dtheta_left = -(le[:ny2, 1] - y0)
            dx_dtheta = np.hstack((dx_dtheta_left, dx_dtheta_right))

        partials["mesh", "sweep"] = np.tile(dx_dtheta * dtan_dtheta, nx)

        nn = nx * ny * 3
        partials["mesh", "in_mesh"][:nn] = 1.0

        nn2 = nx * (ny - 1)
        partials["mesh", "in_mesh"][nn : nn + nn2] = tan_theta
        partials["mesh", "in_mesh"][nn + nn2 :] = -tan_theta


class ShearX(om.ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by shearing the wing in the x direction
    (distributed sweep).

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    xshear[ny] : numpy array
        Distance to translate wing in x direction.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("val", desc="Initial value for x shear.")
        self.options.declare("mesh_shape", desc="Tuple containing mesh shape (nx, ny).")

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]

        self.add_input("xshear", val=val, units="m")
        self.add_input("in_mesh", shape=mesh_shape, units="m")

        self.add_output("mesh", shape=mesh_shape, units="m")

        nx, ny, _ = mesh_shape

        nn = nx * ny
        rows = 3.0 * np.arange(nn)
        cols = np.tile(np.arange(ny), nx)
        val = np.ones(nn)

        self.declare_partials("mesh", "xshear", rows=rows, cols=cols, val=val)

        nn = nx * ny * 3
        rows = np.arange(nn)
        cols = np.arange(nn)
        val = np.ones(nn)

        self.declare_partials("mesh", "in_mesh", rows=rows, cols=cols, val=val)

    def compute(self, inputs, outputs):
        outputs["mesh"][:] = inputs["in_mesh"]
        outputs["mesh"][:, :, 0] += inputs["xshear"]


class Stretch(om.ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by stretching the mesh in spanwise direction to
    reach specified span

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    span : float
        Relative stetch ratio in the spanwise direction.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the stretched aerodynamic surface.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("val", desc="Initial value for span.")
        self.options.declare("mesh_shape", desc="Tuple containing mesh shape (nx, ny).")
        self.options.declare(
            "symmetry", default=False, desc="Flag set to true if surface is reflected about y=0 plane."
        )
        self.options.declare(
            "ref_axis_pos",
            default=0.25,
            desc="Fraction of the chord that is use as the reference axis line",
        )

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]
        self.ref_axis_pos = self.options["ref_axis_pos"]

        self.add_input("span", val=val, units="m")
        self.add_input("in_mesh", shape=mesh_shape, units="m")

        self.add_output("mesh", shape=mesh_shape, units="m")

        nx, ny, _ = mesh_shape
        nn = nx * ny
        rows = 3 * np.arange(nn) + 1
        cols = np.zeros(nn)

        self.declare_partials("mesh", "span", rows=rows, cols=cols)

        # First: x and z on diag is identity.
        nn = nx * ny
        xz_diag = 3 * np.arange(nn)

        # Four columns at le (root, tip) and te (root, tip)
        i_le0 = 1
        i_le1 = ny * 3 - 2
        i_te0 = (nx - 1) * ny * 3 + 1
        i_te1 = nn * 3 - 2

        rows_4c = np.tile(3 * np.arange(nn) + 1, 4)
        cols_4c = np.concatenate([np.tile(i_le0, nn), np.tile(i_le1, nn), np.tile(i_te0, nn), np.tile(i_te1, nn)])

        # Diagonal stripes
        base = 3 * np.arange(1, ny - 1) + 1
        row_dg = np.tile(base, nx) + np.repeat(ny * 3 * np.arange(nx), ny - 2)
        rows_dg = np.tile(row_dg, 2)
        col_dg = np.tile(base, nx)
        cols_dg = np.concatenate([col_dg, col_dg + 3 * ny * (nx - 1)])

        rows = np.concatenate([xz_diag, xz_diag + 2, rows_4c, rows_dg])
        cols = np.concatenate([xz_diag, xz_diag + 2, cols_4c, cols_dg])

        self.declare_partials("mesh", "in_mesh", rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        symmetry = self.options["symmetry"]
        span = inputs["span"][0]
        mesh = inputs["in_mesh"]

        # Set the span along the quarter-chord line
        le = mesh[0]
        te = mesh[-1]
        ref_axis = self.ref_axis_pos * te + (1 - self.ref_axis_pos) * le

        # The user always deals with the full span, so if they input a specific
        # span value and have symmetry enabled, we divide this value by 2.
        if symmetry:
            span /= 2.0

        # Compute the previous span and determine the scalar needed to reach the
        # desired span
        prev_span = ref_axis[-1, 1] - ref_axis[0, 1]
        s = ref_axis[:, 1] / prev_span

        outputs["mesh"][:] = mesh
        outputs["mesh"][:, :, 1] = s * span

    def compute_partials(self, inputs, partials):
        symmetry = self.options["symmetry"]
        span = inputs["span"][0]
        mesh = inputs["in_mesh"]
        nx, ny, _ = mesh.shape

        # Set the span along the quarter-chord line
        le = mesh[0]
        te = mesh[-1]
        ref_axis = self.ref_axis_pos * te + (1 - self.ref_axis_pos) * le

        # The user always deals with the full span, so if they input a specific
        # span value and have symmetry enabled, we divide this value by 2.
        if symmetry:
            span /= 2.0

        # Compute the previous span and determine the scalar needed to reach the
        # desired span
        prev_span = ref_axis[-1, 1] - ref_axis[0, 1]
        s = ref_axis[:, 1] / prev_span

        d_prev_span = -ref_axis[:, 1] / prev_span**2
        d_prev_span_qc0 = np.zeros((ny,))
        d_prev_span_qc1 = np.zeros((ny,))
        d_prev_span_qc0[0] = d_prev_span_qc1[-1] = 1.0 / prev_span

        if symmetry:
            partials["mesh", "span"] = np.tile(0.5 * s, nx)
        else:
            partials["mesh", "span"] = np.tile(s, nx)

        nn = nx * ny * 2
        partials["mesh", "in_mesh"][:nn] = 1.0

        nn2 = nx * ny
        partials["mesh", "in_mesh"][nn : nn + nn2] = np.tile(
            -(1 - self.ref_axis_pos) * span * (d_prev_span - d_prev_span_qc0), nx
        )
        nn3 = nn + nn2 * 2
        partials["mesh", "in_mesh"][nn + nn2 : nn3] = np.tile(
            (1 - self.ref_axis_pos) * span * (d_prev_span + d_prev_span_qc1), nx
        )
        nn4 = nn3 + nn2
        partials["mesh", "in_mesh"][nn3:nn4] = np.tile(-self.ref_axis_pos * span * (d_prev_span - d_prev_span_qc0), nx)
        nn5 = nn4 + nn2
        partials["mesh", "in_mesh"][nn4:nn5] = np.tile(self.ref_axis_pos * span * (d_prev_span + d_prev_span_qc1), nx)

        nn6 = nn5 + nx * (ny - 2)
        partials["mesh", "in_mesh"][nn5:nn6] = (1 - self.ref_axis_pos) * span / prev_span
        partials["mesh", "in_mesh"][nn6:] = self.ref_axis_pos * span / prev_span


class ShearY(om.ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by shearing the wing in the y direction
    (distributed sweep).

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    yshear[ny] : numpy array
        Distance to translate wing in y direction.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("val", desc="Initial value for y shear.")
        self.options.declare("mesh_shape", desc="Tuple containing mesh shape (nx, ny).")

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]

        self.add_input("yshear", val=val, units="m")
        self.add_input("in_mesh", shape=mesh_shape, units="m")

        self.add_output("mesh", shape=mesh_shape, units="m")

        nx, ny, _ = mesh_shape

        nn = nx * ny
        rows = 3.0 * np.arange(nn) + 1
        cols = np.tile(np.arange(ny), nx)
        val = np.ones(nn)

        self.declare_partials("mesh", "yshear", rows=rows, cols=cols, val=val)

        nn = nx * ny * 3
        rows = np.arange(nn)
        cols = np.arange(nn)
        val = np.ones(nn)

        self.declare_partials("mesh", "in_mesh", rows=rows, cols=cols, val=val)

    def compute(self, inputs, outputs):
        outputs["mesh"][:] = inputs["in_mesh"]
        outputs["mesh"][:, :, 1] += inputs["yshear"]


class Dihedral(om.ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by applying dihedral angle. Positive angles up.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    dihedral : float
        Dihedral angle in degrees.
    symmetry : boolean
        Flag set to true if surface is reflected about y=0 plane.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the aerodynamic surface with dihedral angle.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("val", desc="Initial value for dihedral.")
        self.options.declare("mesh_shape", desc="Tuple containing mesh shape (nx, ny).")
        self.options.declare(
            "symmetry", default=False, desc="Flag set to true if surface is reflected about y=0 plane."
        )

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]

        self.add_input("dihedral", val=val, units="deg")
        self.add_input("in_mesh", shape=mesh_shape, units="m")

        self.add_output("mesh", shape=mesh_shape, units="m")

        nx, ny, _ = mesh_shape
        nn = nx * ny
        rows = 3 * np.arange(nn) + 2
        cols = np.zeros(nn)

        self.declare_partials("mesh", "dihedral", rows=rows, cols=cols)

        nn = nx * ny * 3
        n_rows = np.arange(nn)

        if self.options["symmetry"]:
            y_cp = ny * 3 - 2
            te_cols = np.tile(y_cp, nx * (ny - 1))
            te_rows = np.tile(3 * np.arange(ny - 1) + 2, nx) + np.repeat(3 * ny * np.arange(nx), ny - 1)
            se_cols = np.tile(3 * np.arange(ny - 1) + 1, nx)
        else:
            y_cp = 3 * (ny + 1) // 2 - 2
            n_sym = (ny - 1) // 2

            te_row = np.tile(3 * np.arange(n_sym) + 2, 2) + np.repeat([0, 3 * (n_sym + 1)], n_sym)
            te_rows = np.tile(te_row, nx) + np.repeat(3 * ny * np.arange(nx), ny - 1)

            te_col = np.tile(y_cp, n_sym)
            se_col1 = 3 * np.arange(n_sym) + 1
            se_col2 = 3 * np.arange(n_sym) + 4 + 3 * n_sym

            # neat trick: swap columns on reflected side so we can assign in just two operations
            te_cols = np.tile(np.concatenate([te_col, se_col2]), nx)
            se_cols = np.tile(np.concatenate([se_col1, te_col]), nx)

        rows = np.concatenate(([n_rows, te_rows, te_rows]))
        cols = np.concatenate(([n_rows, te_cols, se_cols]))

        self.declare_partials("mesh", "in_mesh", rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        symmetry = self.options["symmetry"]
        dihedral_angle = inputs["dihedral"][0]
        mesh = inputs["in_mesh"]

        # Get the mesh parameters and desired sweep angle
        _, ny, _ = mesh.shape
        le = mesh[0]
        p180 = np.pi / 180
        tan_theta = np.tan(p180 * dihedral_angle)

        # If symmetric, simply vary the z-coord based on the distance from the
        # center of the wing
        if symmetry:
            y0 = le[-1, 1]
            dz = -(le[:, 1] - y0) * tan_theta

        else:
            ny2 = (ny - 1) // 2
            y0 = le[ny2, 1]
            dz_right = (le[ny2:, 1] - y0) * tan_theta
            dz_left = -(le[:ny2, 1] - y0) * tan_theta
            dz = np.hstack((dz_left, dz_right))

        # dz added spanwise.
        outputs["mesh"][:] = mesh
        outputs["mesh"][:, :, 2] += dz

    def compute_partials(self, inputs, partials):
        symmetry = self.options["symmetry"]
        dihedral_angle = inputs["dihedral"][0]
        mesh = inputs["in_mesh"]

        # Get the mesh parameters and desired sweep angle
        nx, ny, _ = mesh.shape
        le = mesh[0]
        p180 = np.pi / 180
        tan_theta = np.tan(p180 * dihedral_angle)
        dtan_dangle = p180 / np.cos(p180 * dihedral_angle) ** 2

        # If symmetric, simply vary the z-coord based on the distance from the
        # center of the wing
        if symmetry:
            y0 = le[-1, 1]
            dz_dtheta = -(le[:, 1] - y0) * dtan_dangle

        else:
            ny2 = (ny - 1) // 2
            y0 = le[ny2, 1]

            ddz_right = (le[ny2:, 1] - y0) * dtan_dangle
            ddz_left = -(le[:ny2, 1] - y0) * dtan_dangle
            dz_dtheta = np.hstack((ddz_left, ddz_right))

        # dz added spanwise.
        partials["mesh", "dihedral"] = np.tile(dz_dtheta, nx)

        nn = nx * ny * 3
        partials["mesh", "in_mesh"][:nn] = 1.0

        nn2 = nx * (ny - 1)
        partials["mesh", "in_mesh"][nn : nn + nn2] = tan_theta
        partials["mesh", "in_mesh"][nn + nn2 :] = -tan_theta


class ShearZ(om.ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by shearing the wing in the z direction
    (distributed sweep).

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    zshear[ny] : numpy array
        Distance to translate wing in z direction.

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh with the new chord lengths.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("val", desc="Initial value for z shear.")
        self.options.declare("mesh_shape", desc="Tuple containing mesh shape (nx, ny).")

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]

        self.add_input("zshear", val=val, units="m")
        self.add_input("in_mesh", shape=mesh_shape, units="m")

        self.add_output("mesh", shape=mesh_shape, units="m")

        nx, ny, _ = mesh_shape

        nn = nx * ny
        rows = 3.0 * np.arange(nn) + 2
        cols = np.tile(np.arange(ny), nx)
        val = np.ones(nn)

        self.declare_partials("mesh", "zshear", rows=rows, cols=cols, val=val)

        nn = nx * ny * 3
        rows = np.arange(nn)
        cols = np.arange(nn)
        val = np.ones(nn)

        self.declare_partials("mesh", "in_mesh", rows=rows, cols=cols, val=val)

    def compute(self, inputs, outputs):
        outputs["mesh"][:] = inputs["in_mesh"]
        outputs["mesh"][:, :, 2] += inputs["zshear"]


class Rotate(om.ExplicitComponent):
    """
    OpenMDAO component that manipulates the mesh by compute rotation matrices given mesh and
    rotation angles in degrees.

    Parameters
    ----------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the initial aerodynamic surface.
    theta_y[ny] : numpy array
        1-D array of rotation angles about y-axis for each wing slice in degrees.
    symmetry : boolean
        Flag set to True if surface is reflected about y=0 plane.
    rotate_x : boolean
        Flag set to True if the user desires the twist variable to always be
        applied perpendicular to the wing (say, in the case of a winglet).

    Returns
    -------
    mesh[nx, ny, 3] : numpy array
        Nodal mesh defining the twisted aerodynamic surface.
    """

    def initialize(self):
        """
        Declare options.
        """
        self.options.declare("val", desc="Initial value for dihedral.")
        self.options.declare("mesh_shape", desc="Tuple containing mesh shape (nx, ny).")
        self.options.declare(
            "symmetry", default=False, desc="Flag set to true if surface is reflected about y=0 plane."
        )
        self.options.declare(
            "rotate_x",
            default=True,
            desc="Flag set to True if the user desires the twist variable to "
            "always be applied perpendicular to the wing (say, in the case of "
            "a winglet).",
        )
        self.options.declare(
            "ref_axis_pos",
            default=0.25,
            desc="Fraction of the chord that is use as the reference axis line",
        )

    def setup(self):
        mesh_shape = self.options["mesh_shape"]
        val = self.options["val"]
        self.ref_axis_pos = self.options["ref_axis_pos"]

        self.add_input("twist", val=val, units="deg")
        self.add_input("in_mesh", shape=mesh_shape, units="m")

        self.add_output("mesh", shape=mesh_shape, units="m")

        nx, ny, _ = mesh_shape
        nn = nx * ny * 3
        rows = np.arange(nn)
        col = np.tile(np.zeros(3), ny) + np.repeat(np.arange(ny), 3)
        cols = np.tile(col, nx)

        self.declare_partials("mesh", "twist", rows=rows, cols=cols)

        row_base = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        col_base = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

        # Diagonal
        nn = nx * ny
        dg_row = np.tile(row_base, nn) + np.repeat(3 * np.arange(nn), 9)
        dg_col = np.tile(col_base, nn) + np.repeat(3 * np.arange(nn), 9)

        # Leading and Trailing edge on diagonal terms.
        row_base_y = np.tile(row_base, ny) + np.repeat(3 * np.arange(ny), 9)
        col_base_y = np.tile(col_base, ny) + np.repeat(3 * np.arange(ny), 9)
        nn2 = 3 * ny
        te_dg_row = np.tile(row_base_y, nx - 1) + np.repeat(nn2 * np.arange(nx - 1), 9 * ny)
        le_dg_col = np.tile(col_base_y, nx - 1)
        le_dg_row = te_dg_row + nn2
        te_dg_col = le_dg_col + 3 * ny * (nx - 1)

        # Leading and Trailing edge off diagonal terms.
        if self.options["symmetry"]:
            row_base_y = np.tile(row_base, ny - 1) + np.repeat(3 * np.arange(ny - 1), 9)
            col_base_y = np.tile(col_base + 3, ny - 1) + np.repeat(3 * np.arange(ny - 1), 9)

            nn2 = 3 * ny
            te_od_row = np.tile(row_base_y, nx) + np.repeat(nn2 * np.arange(nx), 9 * (ny - 1))
            le_od_col = np.tile(col_base_y, nx)
            te_od_col = le_od_col + 3 * ny * (nx - 1)

            rows = np.concatenate([dg_row, le_dg_row, te_dg_row, te_od_row, te_od_row])
            cols = np.concatenate([dg_col, le_dg_col, te_dg_col, le_od_col, te_od_col])

        else:
            n_sym = (ny - 1) // 2

            row_base_y1 = np.tile(row_base, n_sym) + np.repeat(3 * np.arange(n_sym), 9)
            col_base_y1 = np.tile(col_base + 3, n_sym) + np.repeat(3 * np.arange(n_sym), 9)

            row_base_y2 = row_base_y1 + 3 * n_sym + 3
            col_base_y2 = col_base_y1 + 3 * n_sym - 3

            nn2 = 3 * ny

            te_od_row1 = np.tile(row_base_y1, nx) + np.repeat(nn2 * np.arange(nx), 9 * n_sym)
            le_od_col1 = np.tile(col_base_y1, nx)
            te_od_col1 = le_od_col1 + 3 * ny * (nx - 1)
            te_od_row2 = np.tile(row_base_y2, nx) + np.repeat(nn2 * np.arange(nx), 9 * n_sym)
            le_od_col2 = np.tile(col_base_y2, nx)
            te_od_col2 = le_od_col2 + 3 * ny * (nx - 1)

            rows = np.concatenate([dg_row, le_dg_row, te_dg_row, te_od_row1, te_od_row2, te_od_row1, te_od_row2])
            cols = np.concatenate([dg_col, le_dg_col, te_dg_col, le_od_col1, le_od_col2, te_od_col1, te_od_col2])

        self.declare_partials("mesh", "in_mesh", rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        symmetry = self.options["symmetry"]
        rotate_x = self.options["rotate_x"]
        theta_y = inputs["twist"]
        mesh = inputs["in_mesh"]

        te = mesh[-1]
        le = mesh[0]
        ref_axis = self.ref_axis_pos * te + (1 - self.ref_axis_pos) * le

        _, ny, _ = mesh.shape

        if rotate_x:
            # Compute spanwise z displacements along quarter chord
            if symmetry:
                dz_qc = ref_axis[:-1, 2] - ref_axis[1:, 2]
                dy_qc = ref_axis[:-1, 1] - ref_axis[1:, 1]
                theta_x = np.arctan(dz_qc / dy_qc)

                # Prepend with 0 so that root is not rotated
                rad_theta_x = np.append(theta_x, 0.0)
            else:
                root_index = int((ny - 1) / 2)
                dz_qc_left = ref_axis[:root_index, 2] - ref_axis[1 : root_index + 1, 2]
                dy_qc_left = ref_axis[:root_index, 1] - ref_axis[1 : root_index + 1, 1]
                theta_x_left = np.arctan(dz_qc_left / dy_qc_left)
                dz_qc_right = ref_axis[root_index + 1 :, 2] - ref_axis[root_index:-1, 2]
                dy_qc_right = ref_axis[root_index + 1 :, 1] - ref_axis[root_index:-1, 1]
                theta_x_right = np.arctan(dz_qc_right / dy_qc_right)

                # Concatenate thetas
                rad_theta_x = np.concatenate((theta_x_left, np.zeros(1), theta_x_right))

        else:
            rad_theta_x = 0.0

        rad_theta_y = theta_y * np.pi / 180.0

        mats = np.zeros((ny, 3, 3), dtype=type(rad_theta_y[0]))

        cos_rtx = np.cos(rad_theta_x)
        cos_rty = np.cos(rad_theta_y)
        sin_rtx = np.sin(rad_theta_x)
        sin_rty = np.sin(rad_theta_y)

        mats[:, 0, 0] = cos_rty
        mats[:, 0, 2] = sin_rty
        mats[:, 1, 0] = sin_rtx * sin_rty
        mats[:, 1, 1] = cos_rtx
        mats[:, 1, 2] = -sin_rtx * cos_rty
        mats[:, 2, 0] = -cos_rtx * sin_rty
        mats[:, 2, 1] = sin_rtx
        mats[:, 2, 2] = cos_rtx * cos_rty

        outputs["mesh"] = np.einsum("ikj, mij -> mik", mats, mesh - ref_axis) + ref_axis

    def compute_partials(self, inputs, partials):
        symmetry = self.options["symmetry"]
        rotate_x = self.options["rotate_x"]
        theta_y = inputs["twist"]
        mesh = inputs["in_mesh"]

        te = mesh[-1]
        le = mesh[0]
        ref_axis = self.ref_axis_pos * te + (1 - self.ref_axis_pos) * le

        nx, ny, _ = mesh.shape

        if rotate_x:
            # Compute spanwise z displacements along quarter chord
            if symmetry:
                dz_qc = ref_axis[:-1, 2] - ref_axis[1:, 2]
                dy_qc = ref_axis[:-1, 1] - ref_axis[1:, 1]
                theta_x = np.arctan(dz_qc / dy_qc)

                # Prepend with 0 so that root is not rotated
                rad_theta_x = np.append(theta_x, 0.0)

                fact = 1.0 / (1.0 + (dz_qc / dy_qc) ** 2)

                dthx_dq = np.zeros((ny, 3))
                dthx_dq[:-1, 1] = -dz_qc * fact / dy_qc**2
                dthx_dq[:-1, 2] = fact / dy_qc

            else:
                root_index = int((ny - 1) / 2)
                dz_qc_left = ref_axis[:root_index, 2] - ref_axis[1 : root_index + 1, 2]
                dy_qc_left = ref_axis[:root_index, 1] - ref_axis[1 : root_index + 1, 1]
                theta_x_left = np.arctan(dz_qc_left / dy_qc_left)
                dz_qc_right = ref_axis[root_index + 1 :, 2] - ref_axis[root_index:-1, 2]
                dy_qc_right = ref_axis[root_index + 1 :, 1] - ref_axis[root_index:-1, 1]
                theta_x_right = np.arctan(dz_qc_right / dy_qc_right)

                # Concatenate thetas
                rad_theta_x = np.concatenate((theta_x_left, np.zeros(1), theta_x_right))

                fact_left = 1.0 / (1.0 + (dz_qc_left / dy_qc_left) ** 2)
                fact_right = 1.0 / (1.0 + (dz_qc_right / dy_qc_right) ** 2)

                dthx_dq = np.zeros((ny, 3))
                dthx_dq[:root_index, 1] = -dz_qc_left * fact_left / dy_qc_left**2
                dthx_dq[root_index + 1 :, 1] = -dz_qc_right * fact_right / dy_qc_right**2
                dthx_dq[:root_index, 2] = fact_left / dy_qc_left
                dthx_dq[root_index + 1 :, 2] = fact_right / dy_qc_right

        else:
            rad_theta_x = 0.0

        deg2rad = np.pi / 180.0
        rad_theta_y = theta_y * deg2rad

        mats = np.zeros((ny, 3, 3), dtype=type(rad_theta_y[0]))

        cos_rtx = np.cos(rad_theta_x)
        cos_rty = np.cos(rad_theta_y)
        sin_rtx = np.sin(rad_theta_x)
        sin_rty = np.sin(rad_theta_y)

        mats[:, 0, 0] = cos_rty
        mats[:, 0, 2] = sin_rty
        mats[:, 1, 0] = sin_rtx * sin_rty
        mats[:, 1, 1] = cos_rtx
        mats[:, 1, 2] = -sin_rtx * cos_rty
        mats[:, 2, 0] = -cos_rtx * sin_rty
        mats[:, 2, 1] = sin_rtx
        mats[:, 2, 2] = cos_rtx * cos_rty

        dmats_dthy = np.zeros((ny, 3, 3))
        dmats_dthy[:, 0, 0] = -sin_rty * deg2rad
        dmats_dthy[:, 0, 2] = cos_rty * deg2rad
        dmats_dthy[:, 1, 0] = sin_rtx * cos_rty * deg2rad
        dmats_dthy[:, 1, 2] = sin_rtx * sin_rty * deg2rad
        dmats_dthy[:, 2, 0] = -cos_rtx * cos_rty * deg2rad
        dmats_dthy[:, 2, 2] = -cos_rtx * sin_rty * deg2rad

        d_dthetay = np.einsum("ikj, mij -> mik", dmats_dthy, mesh - ref_axis)
        partials["mesh", "twist"] = d_dthetay.flatten()

        nn = nx * ny * 9
        partials["mesh", "in_mesh"][:nn] = np.tile(mats.flatten(), nx)

        # Quarter chord direct contribution.
        eye = np.tile(np.eye(3).flatten(), ny).reshape(ny, 3, 3)
        d_qch = (eye - mats).flatten()

        nqc = ny * 9
        partials["mesh", "in_mesh"][:nqc] += (1 - self.ref_axis_pos) * d_qch
        partials["mesh", "in_mesh"][nn - nqc : nn] += self.ref_axis_pos * d_qch

        if rotate_x:
            dmats_dthx = np.zeros((ny, 3, 3))
            dmats_dthx[:, 1, 0] = cos_rtx * sin_rty
            dmats_dthx[:, 1, 1] = -sin_rtx
            dmats_dthx[:, 1, 2] = -cos_rtx * cos_rty
            dmats_dthx[:, 2, 0] = sin_rtx * sin_rty
            dmats_dthx[:, 2, 1] = cos_rtx
            dmats_dthx[:, 2, 2] = -sin_rtx * cos_rty

            d_dthetax = np.einsum("ikj, mij -> mik", dmats_dthx, mesh - ref_axis)
            d_dq = np.einsum("ijk, jm -> ijkm", d_dthetax, dthx_dq)

            d_dq_flat = d_dq.flatten()

            del_n = nn - 9 * ny
            nn2 = nn + del_n
            nn3 = nn2 + del_n
            partials["mesh", "in_mesh"][nn:nn2] = (1 - self.ref_axis_pos) * d_dq_flat[-del_n:]
            partials["mesh", "in_mesh"][nn2:nn3] = self.ref_axis_pos * d_dq_flat[:del_n]

            # Contribution back to main diagonal.
            del_n = 9 * ny
            partials["mesh", "in_mesh"][:nqc] += (1 - self.ref_axis_pos) * d_dq_flat[:del_n]
            partials["mesh", "in_mesh"][nn - nqc : nn] += self.ref_axis_pos * d_dq_flat[-del_n:]

            # Quarter chord direct contribution.
            d_qch_od = np.tile(d_qch.flatten(), nx - 1)
            partials["mesh", "in_mesh"][nn:nn2] += (1 - self.ref_axis_pos) * d_qch_od
            partials["mesh", "in_mesh"][nn2:nn3] += self.ref_axis_pos * d_qch_od

            # off-off diagonal pieces
            if symmetry:
                d_dq_flat = d_dq[:, :-1, :, :].flatten()

                del_n = nn - 9 * nx
                nn4 = nn3 + del_n
                partials["mesh", "in_mesh"][nn3:nn4] = -(1 - self.ref_axis_pos) * d_dq_flat
                nn5 = nn4 + del_n
                partials["mesh", "in_mesh"][nn4:nn5] = -self.ref_axis_pos * d_dq_flat

            else:
                d_dq_flat1 = d_dq[:, :root_index, :, :].flatten()
                d_dq_flat2 = d_dq[:, root_index + 1 :, :, :].flatten()

                del_n = nx * root_index * 9
                nn4 = nn3 + del_n
                partials["mesh", "in_mesh"][nn3:nn4] = -(1 - self.ref_axis_pos) * d_dq_flat1
                nn5 = nn4 + del_n
                partials["mesh", "in_mesh"][nn4:nn5] = -(1 - self.ref_axis_pos) * d_dq_flat2
                nn6 = nn5 + del_n
                partials["mesh", "in_mesh"][nn5:nn6] = -self.ref_axis_pos * d_dq_flat1
                nn7 = nn6 + del_n
                partials["mesh", "in_mesh"][nn6:nn7] = -self.ref_axis_pos * d_dq_flat2
