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


def test_interpolation_on_grid():
    """
    Test the interpolation on a 3D grid.

    This function initializes an interpolation object, creates a 3D grid of random values,
    and applies sinusoidal function to populate the grid. It then interpolates these values
    using the `interp` class and compares the interpolated values with the ground truth
    obtained by directly applying the sinusoidal function to the interpolated positions.
    An assertion is used to check if the interpolation error is within the specified tolerance.

    Attributes:
    tol (float): Tolerance level for the difference between interpolated and ground truth values.
    length (int): Length of each dimension in the grid.
    dx (float): Differential step to scale the grid positions.
    """
    for centering in [True, False]:
        for num_points in [4, 6, 8]:
            tol = 1e-9
            channels = 25
            interpolation = interp(
                num_points=num_points,
                max_degree=num_points // 2,
                num_channels=channels,
                learnable=False,
                align_corners=centering,
            )
            length = 10
            dx = 0.01

            # Initializing a tensor of random values to represent the grid
            x = torch.rand(2, channels, length, length, length)

            # Preparing input positions for the sinusoidal function
            input_positions = torch.zeros(length, length, length, 3)
            for i in range(x.shape[2]):
                for j in range(x.shape[3]):
                    for k in range(x.shape[4]):
                        input_positions[i, j, k] = torch.tensor([i, j, k])
                        pos = dx * np.array([i, j, k])
                        x[:, :, i, j, k] = sinusoidal_function(*pos)

            # Perform interpolation and measure time taken
            time1 = time.time()
            interpolated = interpolation(x)
            interpolated_old = interpolation.non_vector_implementation(x)
            print(f"Time taken for interpolation: {(time.time() - time1):.2f} sec")
            positions = interpolation.get_postion(x)

            # Preparing ground truth for comparison
            ghosts = int(math.ceil(num_points / 2))
            shape = x.shape
            ground_truth = torch.zeros(
                shape[0],
                shape[1],
                (shape[2] - 2 * ghosts) * 2 + 2,
                (shape[3] - 2 * ghosts) * 2 + 2,
                (shape[4] - 2 * ghosts) * 2 + 2,
            )

            # Applying sinusoidal function to the interpolated positions

            shape = ground_truth.shape
            # Perform interpolation
            for i in range(shape[2]):
                for j in range(shape[3]):
                    for k in range(shape[4]):
                        pos = dx * (positions[i, j, k])
                        ground_truth[:, :, i, j, k] = sinusoidal_function(*pos)

            # Comparing interpolated and ground truth values
            # assert((torch.mean(torch.abs(interpolated - ground_truth))))
            assert (torch.mean(torch.abs(interpolated - ground_truth))) < tol

            # Comparing old and new interpolation
            assert torch.mean(torch.abs(interpolated_old - interpolated)) < tol


if __name__ == "__main__":
    test_interpolation_stencils()
    test_interpolation_on_grid()
