from RadiShPICR.ConstraintBasedRelativity.particle_sources import source_terms_at_point


def mass_density_at_point(
    particles, A_at_point, radial_coordinate, grid, shape_mode=None, *, A_values,
):
    """Use the completed metric to locate each particle once."""
    return source_terms_at_point(
        particles, A_at_point, radial_coordinate, grid, shape_mode, A_values=A_values,
    )[0]
