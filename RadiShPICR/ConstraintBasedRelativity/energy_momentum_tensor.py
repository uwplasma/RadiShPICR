from RadiShPICR.ConstraintBasedRelativity.particle_sources import source_terms_at_point


def Sr_at_point(
    particles, A_at_point, radial_coordinate, grid, shape_mode=None, *, A_values,
):
    """Use the completed metric to locate each particle once."""
    return source_terms_at_point(
        particles, A_at_point, radial_coordinate, grid, shape_mode, A_values=A_values,
    )[3]


def Srr_at_point(
    particles, A_at_point, radial_coordinate, grid, shape_mode=None, *, A_values,
):
    """Use the completed metric to locate each particle once."""
    return source_terms_at_point(
        particles, A_at_point, radial_coordinate, grid, shape_mode, A_values=A_values,
    )[2]
