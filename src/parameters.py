"""Physical parameters for the axially moving web model."""

# Span length [m]
span_length_m = 1.0

# Rectangular cross-section width [m]
width_m = 0.5

# Rectangular cross-section thickness [m]
thickness_m = 0.0002

# Linear mass density, rhoA [kg/m]
linear_density_kg_per_m = 0.08

# Young's modulus [Pa]
youngs_modulus_pa = 2.1e9

# Nominal initial web tension, T0 [N]
initial_tension_n = 200.0

# Rectangular cross-sectional area, A = b*h [m^2]
cross_sectional_area_m2 = width_m * thickness_m

# Rectangular second moment of area, I = b*h^3/12 [m^4]
second_moment_of_area_m4 = width_m * thickness_m**3 / 12.0

# Flexural rigidity, EI [N*m^2]
flexural_rigidity_nm2 = youngs_modulus_pa * second_moment_of_area_m4
