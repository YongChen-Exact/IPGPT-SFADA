import numpy
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import (binary_erosion, distance_transform_edt,
                                      generate_binary_structure)


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(
            voxelspacing, result.ndim)
        voxelspacing = numpy.asarray(voxelspacing, dtype=numpy.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    footprint = generate_binary_structure(result.ndim, connectivity)
    result_border = result ^ binary_erosion(
        result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(
        reference, structure=footprint, iterations=1)
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    return sds


def hd(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(
        result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(
        reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def hd_fast(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(
        result, reference, voxelspacing, connectivity).max()
    hd2 = __surface_distances(
        reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def hd95(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = numpy.percentile(numpy.hstack((hd1, hd2)), 95)
    return hd95


def hd95_fast(result, reference, voxelspacing=None, connectivity=1):
    hd1 = __surface_distances(
        result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(
        reference, result, voxelspacing, connectivity)
    hd95 = numpy.percentile(numpy.hstack((hd1, hd2)), 95)
    return hd95


def assd(result, reference, voxelspacing=None, connectivity=1):
    assd = numpy.mean((__surface_distances(result, reference, voxelspacing, connectivity),
                       __surface_distances(reference, result, voxelspacing, connectivity)))
    return assd


def asd(result, reference, voxelspacing=None, connectivity=1):
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd


def asd_fast(result, reference, voxelspacing=None, connectivity=1):
    sds = __surface_distances(
        result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd


def nsd(result, reference, voxelspacing=None, tolerance_mm=2):
    nsd = compute_surface_dice_at_tolerance(
        compute_surface_distances(reference, result, voxelspacing), tolerance_mm)
    return nsd
