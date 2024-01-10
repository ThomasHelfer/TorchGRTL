from GeneralRelativity.Interpolation import *  # Import necessary functions from Interpolation module


def sinusoidal_function(x, y, z):
    """
    A sinusoidal function of three variables x, y, and z.
    """
    return np.sin(x) * np.sin(y) * np.sin(z)


def test_interpolation_stencils():
    """
    Tests the interpolation stencils by comparing interpolated values with actual values.

    This function generates a 3D grid of points, computes interpolation stencils, and then
    uses them to interpolate the value of a sinusoidal function at a specified point. It compares the
    interpolated value with the actual value of the function at that point.

    The comparison is made for a sinusoidal function to validate the accuracy of the interpolation.
    """
    # Define a random point at which to perform the interpolation
    pnt = [0.3, 0.2, 0.72]
    tol = 1e-10  # Tolerance for comparing interpolated and actual value

    dx = 1e-3  # Grid spacing
    interp_point = np.array([0.5, 0.5, 0.5])  # Reference interpolation point
    num_points = 6  # Number of points in each dimension
    max_degree = 4  # Maximum degree for polynomial interpolation

    # Generate 3D meshgrid for interpolation
    half = int(np.floor(float(max_degree) / 2.0))
    coarse_grid_x, coarse_grid_y, coarse_grid_z = np.meshgrid(
        (pnt[0] - dx * interp_point[0]) * np.ones(num_points)
        + dx * np.arange(0 - half, num_points - half),
        (pnt[1] - dx * interp_point[1]) * np.ones(num_points)
        + dx * np.arange(0 - half, num_points - half),
        (pnt[2] - dx * interp_point[2]) * np.ones(num_points)
        + dx * np.arange(0 - half, num_points - half),
    )

    # Flatten the grid points for ease of computation
    coarse_grid_points_index = np.vstack(
        [coarse_grid_x.ravel(), coarse_grid_y.ravel(), coarse_grid_z.ravel()]
    ).T

    # Calculate interpolation stencils
    vecvals, _ = calculate_stencils(interp_point, num_points, max_degree)

    # Evaluate the sinusoidal function on the coarse grid points and flatten the result
    coarse_values = sinusoidal_function(
        coarse_grid_x, coarse_grid_y, coarse_grid_z
    ).ravel()

    # Interpolate the value at the specified point using the calculated stencils
    interpolated_value = vecvals @ coarse_values

    # Calculate the ground truth value of the sinusoidal function at the point
    ground_truth = sinusoidal_function(*pnt)

    # Compute the absolute error between interpolated and ground truth values
    error = np.abs(ground_truth - interpolated_value)

    # Assert that the error is within the specified tolerance
    assert error < tol, f"Interpolation error {error} exceeds tolerance {tol}"


if __name__ == "__main__":
    test_interpolation_stencils()
