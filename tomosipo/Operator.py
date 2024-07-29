import numpy as np
import tomosipo as ts
from tomosipo.Data import Data
from tomosipo.astra import (
    create_astra_projector,
    direct_fp,
    direct_bp,
)
from tomosipo.geometry import ConeGeometry


def to_astra_compatible_operator_geometry(vg, pg):
    """Convert volume vector geometry to volume geometry (if necessary)

    ASTRA does not support arbitrarily oriented volume geometries. If
    `vg` is a VolumeVectorGeometry, we rotate and translate both `vg`
    and `pg` such that `vg` is axis-aligned, and positioned on the
    origin, which makes it ASTRA-compatible.

    Parameters
    ----------
    vg:
        volume geometry
    pg:
        projection geometry

    Returns
    -------
    (VolumeGeometry, ProjectionGeometry)
        A non-vector volume geometry centered on the origin and its
        corresponding projection geometry.

    """
    if isinstance(vg, ts.geometry.VolumeGeometry):
        return (vg, pg)

    if not isinstance(vg, ts.geometry.VolumeVectorGeometry):
        raise TypeError(f"Expected volume geometry. Got {type(vg)}. ")

    vg = vg.to_vec()
    # Change perspective *without* changing the voxel volume.
    P = ts.from_perspective(
        pos=vg.pos,
        w=vg.w / ts.vector_calc.norm(vg.w)[:, None],
        v=vg.v / ts.vector_calc.norm(vg.v)[:, None],
        u=vg.u / ts.vector_calc.norm(vg.u)[:, None],
    )
    # Move vg to perspective:
    vg = P * vg
    pg = P * pg.to_vec()

    # Assert that vg is now axis-aligned and positioned on the origin:
    voxel_size = vg.voxel_size
    assert np.allclose(vg.pos, np.array([0, 0, 0]))
    assert np.allclose(vg.w, voxel_size[0] * np.array([1, 0, 0]))
    assert np.allclose(vg.v, voxel_size[1] * np.array([0, 1, 0]))
    assert np.allclose(vg.u, voxel_size[2] * np.array([0, 0, 1]))

    axis_aligned_vg = ts.volume(shape=vg.shape, pos=0, size=vg.size)

    return axis_aligned_vg, pg


def operator(
    volume_geometry,
    projection_geometry,
    voxel_supersampling=1,
    detector_supersampling=1,
    additive=False,
    backend='toolbox'
):
    """Create a new tomographic operator

    Parameters:
    -----------
    volume_geometry: `VolumeGeometry`
        The domain of the operator.

    projection_geometry:  `ProjectionGeometry`
        The range of the operator.

    voxel_supersampling: `int` (optional)
        Specifies the amount of voxel supersampling, i.e., how
        many (one dimension) subvoxels are generated from a single
        parent voxel. The default is 1.

    detector_supersampling: `int` (optional)
        Specifies the amount of detector supersampling, i.e., how
        many rays are cast per detector. The default is 1.

    additive: `bool` (optional)
        Specifies whether the operator should overwrite its range
        (forward) and domain (transpose). When `additive=True`,
        the operator adds instead of overwrites. The default is
        `additive=False`.

    backend: `str` (optional)
        If 'toolbox' will use the ASTRA Toolbox as backend for projection
        and reprojection, and return an `Operator`. If 'kernelkit' uses
        ASTRA KernelKit and will return a compatible `KernelkitOperator`.

    Returns
    -------
    Operator
        A linear tomographic projection operator
    """
    if backend == 'toolbox':
        return Operator(
            volume_geometry,
            projection_geometry,
            voxel_supersampling=voxel_supersampling,
            detector_supersampling=detector_supersampling,
            additive=additive,
        )
    elif backend == 'kernelkit':
        # TODO(Adriaan): would it be possible to initizialize, e.g., FBP, with
        #   a parameter for the backend?
        return KernelkitOperator(
            volume_geometry,
            projection_geometry,
            additive=additive,
        )
    else:
        raise ValueError("`backend` must be either 'toolbox' or 'kernelkit'.")


def _to_link(geometry, x):
    if isinstance(x, Data):
        return x.link
    else:
        return ts.link(geometry, x)


class Operator:
    """A linear tomographic projection operator

    An operator describes and computes the projection from a volume onto a
    projection geometry.
    """

    def __init__(
        self,
        volume_geometry,
        projection_geometry,
        voxel_supersampling=1,
        detector_supersampling=1,
        additive=False,
    ):
        """Create a new tomographic operator

        Parameters
        ----------
        volume_geometry: `VolumeGeometry`
            The domain of the operator.

        projection_geometry:  `ProjectionGeometry`
            The range of the operator.

        voxel_supersampling: `int` (optional)
            Specifies the amount of voxel supersampling, i.e., how
            many (one dimension) subvoxels are generated from a single
            parent voxel. The default is 1.

        detector_supersampling: `int` (optional)
            Specifies the amount of detector supersampling, i.e., how
            many rays are cast per detector. The default is 1.

        additive: `bool` (optional)
            Specifies whether the operator should overwrite its range
            (forward) and domain (transpose). When `additive=True`,
            the operator adds instead of overwrites. The default is
            `additive=False`.

        """
        super(Operator, self).__init__()
        self.volume_geometry = volume_geometry
        self.projection_geometry = projection_geometry

        vg, pg = to_astra_compatible_operator_geometry(
            volume_geometry, projection_geometry
        )
        self.astra_compat_vg = vg
        self.astra_compat_pg = pg

        self.astra_projector = create_astra_projector(
            self.astra_compat_vg,
            self.astra_compat_pg,
            voxel_supersampling=voxel_supersampling,
            detector_supersampling=detector_supersampling,
        )
        self.additive = additive
        self._transpose = BackprojectionOperator(self)

    def _fp(self, volume, out=None):
        vlink = _to_link(self.astra_compat_vg, volume)

        if out is not None:
            plink = _to_link(self.astra_compat_pg, out)
        else:
            if self.additive:
                plink = vlink.new_zeros(self.range_shape)
            else:
                plink = vlink.new_empty(self.range_shape)

        direct_fp(self.astra_projector, vlink, plink, additive=self.additive)

        if isinstance(volume, Data):
            return ts.data(self.projection_geometry, plink.data)
        else:
            return plink.data

    def _bp(self, projection, out=None):
        """Apply backprojection

        :param projection: `np.array` or `Data`
            An input projection dataset. If a numpy array, the shape
            must match the operator geometry. If the projection dataset is
            an instance of `Data`, its geometry must match the
            operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A volume dataset on which the projection dataset has been
            backprojected.
        :rtype: `Data`

        """
        plink = _to_link(self.astra_compat_pg, projection)

        if out is not None:
            vlink = _to_link(self.astra_compat_vg, out)
        else:
            if self.additive:
                vlink = plink.new_zeros(self.domain_shape)
            else:
                vlink = plink.new_empty(self.domain_shape)

        direct_bp(
            self.astra_projector,
            vlink,
            plink,
            additive=self.additive,
        )

        if isinstance(projection, Data):
            return ts.data(self.volume_geometry, vlink.data)
        else:
            return vlink.data

    def __call__(self, volume, out=None):
        """Apply operator

        :param volume: `np.array` or `Data`
            An input volume. If a numpy array, the shape must match
            the operator geometry. If the input volume is an instance
            of `Data`, its geometry must match the operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A projection dataset on which the volume has been forward
            projected.
        :rtype: `Data`

        """
        return self._fp(volume, out)

    def transpose(self):
        """Return backprojection operator"""
        return self._transpose

    @property
    def T(self):
        """The transpose operator

        This property returns the transpose (backprojection) operator.
        """
        return self.transpose()

    @property
    def domain(self):
        """The domain (volume geometry) of the operator"""
        return self.volume_geometry

    @property
    def range(self):
        """The range (projection geometry) of the operator"""
        return self.projection_geometry

    @property
    def domain_shape(self):
        """The expected shape of the input (volume) data"""
        return ts.links.geometry_shape(self.astra_compat_vg)

    @property
    def range_shape(self):
        """The expected shape of the output (projection) data"""
        return ts.links.geometry_shape(self.astra_compat_pg)


class BackprojectionOperator:
    """Transpose of the Forward operator

    The idea of having a dedicated class for the backprojection
    operator, which just saves a link to the "real" operator has
    been shamelessly ripped from OpTomo.

    We have the following property:

    >>> import tomosipo as ts
    >>> vg = ts.volume(shape=10)
    >>> pg = ts.parallel(angles=10, shape=10)
    >>> A = ts.operator(vg, pg)
    >>> A.T is A.T.T.T
    True

    It is nice that we do not allocate a new object every time we use
    `A.T`. If we did, users might save the transpose in a separate
    variable for 'performance reasons', writing

    >>> A = ts.operator(vg, pg)
    >>> A_T = A.T

    This is a waste of time.
    """

    def __init__(
        self,
        parent,
    ):
        """Create a new tomographic operator"""
        super(BackprojectionOperator, self).__init__()
        self.parent = parent

    def __call__(self, projection, out=None):
        """Apply operator

        :param projection: `np.array` or `Data`
            An input projection. If a numpy array, the shape must match
            the operator geometry. If the input volume is an instance
            of `Data`, its geometry must match the operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A projection dataset on which the volume has been forward
            projected.
        :rtype: `Data`

        """
        return self.parent._bp(projection, out)

    def transpose(self):
        """Return forward projection operator"""
        return self.parent

    @property
    def T(self):
        """The transpose of the backprojection operator

        This property returns the transpose (forward projection) operator.
        """
        return self.transpose()

    @property
    def domain(self):
        """The domain (projection geometry) of the operator"""
        return self.parent.range

    @property
    def range(self):
        """The range (volume geometry) of the operator"""
        return self.parent.domain

    @property
    def domain_shape(self):
        """The expected shape of the input (projection) data"""
        return self.parent.range_shape

    @property
    def range_shape(self):
        """The expected shape of the output (volume) data"""
        return self.parent.domain_shape


class KernelkitOperator:
    """A linear tomographic projection using ASTRA KernelKit projectors

    An operator describes and computes the projection from a volume onto a
    projection geometry.
    """

    def __init__(
        self,
        volume_geometry,
        projection_geometry,
        additive=False,
    ):
        """Create a new tomographic operator

        Parameters
        ----------
        volume_geometry: `VolumeGeometry`
            The domain of the operator.

        projection_geometry:  `ProjectionGeometry`
            The range of the operator.

        additive: `bool` (optional)
            Specifies whether the operator should overwrite its range
            (forward) and domain (transpose). When `additive=True`,
            the operator adds instead of overwrites. The default is
            `additive=False`.

        Notes
        -----
            The operator communicates Tomosipo geometries towards
            the world, but internally converts them to KernelKit geometries.
        """
        # TODO(Adriaan): better way to import without a dependency?
        import kernelkit as kk

        super(KernelkitOperator, self).__init__()

        # convert geometries
        # TODO(Adriaan): should I also use "ASTRA compat" geoms here?
        kk_vg = self._to_kernelkit_volume_geometry(volume_geometry)
        kk_pg = self._to_kernelkit_scan_geometry(projection_geometry)

        # set up projectors and use KernelKit operator
        fp = kk.ForwardProjector()
        fp.volume_geometry = kk_vg
        fp.projection_geometry = kk_pg
        bp = kk.BackProjector()
        bp.volume_geometry = kk_vg
        bp.projection_geometry = kk_pg
        self.kernelkit_op = kk.ProjectorOperator(fp, bp)

        # TODO(Adriaan): add setters that handle/avoid changes to geometries
        self.additive = additive
        self.volume_geometry = volume_geometry
        self.projection_geometry = projection_geometry
        self._transpose = BackprojectionOperator(self)

    def _to_kernelkit_volume_geometry(self, volume_geometry):
        import kernelkit as kk

        vg = kk.VolumeGeometry(
            shape=volume_geometry.shape,
            extent_min=[volume_geometry.extent[i][0] for i in range(3)],
            extent_max=[volume_geometry.extent[i][1] for i in range(3)],
            voxel_size=volume_geometry.voxel_size,
        )
        vg.check_consistency()
        return vg

    def _to_kernelkit_scan_geometry(self, projection_geometry):
        import kernelkit as kk

        if not projection_geometry.is_cone:
            raise NotImplementedError("Other geometries than conebeam are "
                                      "not yet implemented. Please file "
                                      "an issue if you need one.")
        assert isinstance(projection_geometry, ConeGeometry), (
            "Unexpected geometry type."
        )

        pix_h = projection_geometry.det_size[0] / projection_geometry.det_shape[0]
        pix_w = projection_geometry.det_size[1] / projection_geometry.det_shape[1]
        det = kk.Detector(
            rows=projection_geometry.det_shape[0],
            cols=projection_geometry.det_shape[1],
            pixel_height=pix_h,
            pixel_width=pix_w
        )

        scan = [kk.ProjectionGeometry(
            source_position=projection_geometry.src_pos[i],
            detector_position=projection_geometry.det_pos[i],
            u=(projection_geometry.det_u[i] / np.linalg.norm(projection_geometry.det_u[i])),
            v=(projection_geometry.det_v[i] / np.linalg.norm(projection_geometry.det_v[i])),
            detector=det,
            beam=kk.Beam.CONE
        ) for i in range(projection_geometry.num_angles)]

        return scan

    @staticmethod
    def _to_link(shape, x):
        """Makes a CuPy array if necessary."""
        if isinstance(x, Data):
            return x.link
        else:
            # We don't use ts.link because it requires passing a geometry,
            # and these take fixed assumptions on the shape of the data.
            return ts.links.cupy.CupyLink(shape, x)

    def _fp(self, volume, out=None):
        vlink = self._to_link(self.domain_shape, volume)

        if out is not None:
            plink = self._to_link(self.range_shape, out)
        else:
            if self.additive:
                plink = vlink.new_zeros(self.range_shape)
            else:
                plink = vlink.new_empty(self.range_shape)

        self.kernelkit_op(vlink.data,
                          out=plink.data,
                          additive=self.additive)

        if isinstance(volume, Data):
            return ts.data(self.projection_geometry, plink.data)
        else:
            return plink.data

    def _bp(self, projection, out=None):
        """Apply backprojection

        :param projection: `np.array` or `Data`
            An input projection dataset. If a numpy array, the shape
            must match the operator geometry. If the projection dataset is
            an instance of `Data`, its geometry must match the
            operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A volume dataset on which the projection dataset has been
            backprojected.
        :rtype: `Data`

        """
        plink = self._to_link(self.range_shape, projection)

        if out is not None:
            vlink = self._to_link(self.domain_shape, out)
        else:
            if self.additive:
                vlink = plink.new_zeros(self.domain_shape)
            else:
                vlink = plink.new_empty(self.domain_shape)

        self.kernelkit_op.T(
            plink.data,
            out=vlink.data,
            additive=self.additive)

        if isinstance(projection, Data):
            return ts.data(self.volume_geometry, vlink.data)
        else:
            return vlink.data

    def __call__(self, volume, out=None):
        """Apply operator

        :param volume: `np.array` or `Data`
            An input volume. If a numpy array, the shape must match
            the operator geometry. If the input volume is an instance
            of `Data`, its geometry must match the operator geometry.
        :param out: `np.array` or `Data` (optional)
            An optional output value. If a numpy array, the shape must
            match the operator geometry. If the out parameter is an
            instance of of `Data`, its geometry must match the
            operator geometry.
        :returns:
            A projection dataset on which the volume has been forward
            projected.
        :rtype: `Data`

        """
        return self._fp(volume, out)

    def transpose(self):
        """Return backprojection operator"""
        return self._transpose

    @property
    def T(self):
        """The transpose operator

        This property returns the transpose (backprojection) operator.
        """
        return self.transpose()

    @property
    def domain(self):
        """The domain (volume geometry) of the operator"""
        return self.volume_geometry

    @property
    def range(self):
        """The range (projection geometry) of the operator"""
        return self.projection_geometry

    @property
    def domain_shape(self):
        """The expected shape of the input (volume) data"""
        return self.kernelkit_op.domain_shape

    @property
    def range_shape(self):
        """The expected shape of the output (projection) data"""
        return self.kernelkit_op.range_shape
