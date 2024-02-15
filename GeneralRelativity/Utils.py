import numpy as np
from tqdm.auto import tqdm, trange
import glob
import torch


class TensorDict:
    def __init__(self, tensor: torch.tensor, keys: list):
        """
        Initialize the TensorDict class with a tensor and corresponding keys.
        tensor: A PyTorch tensor, ideally with the last dimension equal to the number of keys.
        keys: A list of keys corresponding to the tensor's last dimension.
        """
        self.num_index = 4
        if tensor.shape[self.num_index] != len(keys):
            raise ValueError(
                f"The number of keys must match the size of the tensor's {self.num_index} dimension."
            )
        self.tensor = tensor
        self.device = tensor.device
        self.keys = keys
        self.key_to_index = {key: i for i, key in enumerate(keys)}

    def __getitem__(self, key: list) -> torch.tensor:
        """
        Retrieve a slice of the tensor corresponding to the given key.
        For keys like 'h', construct a symmetric tensor from elements 'h11', 'h12', etc.
        For keys like 'Gamma', construct a vector from elements 'Gamma1', 'Gamma2', 'Gamma3'.
        """
        # Dealing with metric cases
        # Returns symmetric metric h_ij with shape [batch,xdim,ydim,zdim,i,j]
        # if 1st derivative terms are handed it returns d_x h_ij with shape [batch,xdim,ydim,zdim,i,j,x]
        # for 2nd derivative it returns d_x d_y h_ij with shape [batch,xdim,ydim,zdim,i,j,x,y]
        if key == "h" or key == "A":
            h_tensor = torch.zeros(
                self.tensor.shape[: self.num_index]
                + (3, 3)
                + self.tensor.shape[(self.num_index + 1) :],
                dtype=self.tensor.dtype,
                device=self.device,
            )
            for i in range(3):
                for j in range(i, 3):
                    ij_key = f"{key}{1+i}{1+j}"
                    if ij_key in self.key_to_index:
                        # This is for dealing with 2nd derivatives (and making sure derivatives indices are last)
                        if len(self.tensor.shape[(self.num_index + 1) :]) == 2:
                            for k in range(3):
                                for l in range(3):
                                    # making sure that the tensor is symmetric in ij
                                    h_tensor[..., i, j, k, l] = h_tensor[
                                        ..., j, i, k, l
                                    ] = self.tensor.index_select(
                                        self.num_index,
                                        torch.tensor([self.key_to_index[ij_key]]).to(
                                            self.device
                                        ),
                                    )[
                                        ..., 0, k, l
                                    ]
                        # This is for dealing with 1st derivatives (and making sure derivatives indices are last)
                        elif len(self.tensor.shape[(self.num_index + 1) :]) == 1:
                            for k in range(3):
                                # making sure that the tensor is symmetric in ij
                                h_tensor[..., i, j, k] = h_tensor[
                                    ..., j, i, k
                                ] = self.tensor.index_select(
                                    self.num_index,
                                    torch.tensor([self.key_to_index[ij_key]]).to(
                                        self.device
                                    ),
                                )[
                                    ..., 0, k
                                ]
                        # reference term
                        else:
                            # making sure that the tensor is symmetric in ij
                            h_tensor[..., i, j] = h_tensor[..., j, i] = self.tensor[
                                ..., self.key_to_index[ij_key]
                            ]
            return h_tensor
        # Dealing with vector cases
        # Returns vector v_i with shape [batch,xdim,ydim,zdim,i]
        # if 1st derivative terms are handed it returns d_x v_i with shape [batch,xdim,ydim,zdim,i,x]
        # for 2nd derivative it returns d_x d_y v_i with shape [batch,xdim,ydim,zdim,i,x,y]
        elif (key == "Gamma") or (key == "shift") or (key == "B"):
            indices = [self.key_to_index[f"{key}{i+1}"] for i in range(3)]
            return torch.index_select(
                self.tensor, self.num_index, torch.tensor(indices).to(self.device)
            )
        # Dealing with scalar cases
        # Returns vector s with shape [batch,xdim,ydim,zdim]
        # if 1st derivative terms are handed it returns d_x s with shape [batch,xdim,ydim,zdim,x]
        # for 2nd derivative it returns d_x d_y swith shape [batch,xdim,ydim,zdim,x,y]
        elif key in self.key_to_index:
            return torch.index_select(
                self.tensor,
                self.num_index,
                torch.tensor([self.key_to_index[key]]).to(self.device),
            ).squeeze(self.num_index)
        else:
            raise KeyError(f"Key '{key}' not found.")


def get_binary_data(filename: str, num_variables: int) -> list:
    """
    Reads binary data from a file and returns it as a list of numpy arrays.

    Each array in the list represents a chunk of data from the file. The function
    assumes that each element in the file is a 64-bit double (8 bytes).

    Parameters:
    filename (str): The path to the binary file to be read.
    num_variables (int): The number of variables (doubles) to read in each chunk.

    Returns:
    list: A list of numpy arrays, each containing 'num_variables' elements read
          from the file.
    """
    data_list = []
    length = 0

    with open(filename, "rb") as f:
        while True:
            # Try reading a chunk of data. If unsuccessful, break.
            chunk = f.read(num_variables * 8)  # 8 bytes per double
            if not chunk:
                break

            # Convert the chunk to a numpy array and append it to the list
            array = np.frombuffer(chunk, dtype=np.float64)
            data_list.append(array)
            length += 1
    return data_list


def get_many_binary_files(filenames: str, num_variables: int) -> np.ndarray:
    """
    Reads multiple binary files specified by a pattern and returns their data as a numpy array.

    Each element in the array represents data read from one file. The function assumes that each
    element in the files is a 64-bit double.

    Parameters:
    filenames (str): A pattern string to match filenames. Can include wildcard characters.
    num_variables (int): The number of variables (doubles) to read in each chunk from each file.

    Returns:
    np.ndarray: A numpy array containing the data read from all matched files.
    """
    FilenamesList = glob.glob(filenames)
    tmp = []
    print("Loading data")
    for i in tqdm(FilenamesList):
        tmp += get_binary_data(i, num_variables)
    return np.array(tmp)


def get_many_binary_files_as_array(filenames: str, num_variables: int) -> np.ndarray:
    """
    Reads multiple binary files specified by a pattern and returns their data as a list
    of numpy arrays.

    Each array in the list represents the data from one file. The function assumes that each
    element in the files is a 64-bit double.

    Parameters:
    filenames (str): A pattern string to match filenames. Can include wildcard characters.
    num_variables (int): The number of variables (doubles) to read in each chunk from each file.

    Returns:
    list: A list of numpy arrays, each containing the data read from individual files.
    """
    FilenamesList = glob.glob(filenames)
    tmp = []
    print("Loading data")
    for i in tqdm(FilenamesList):
        tmp.append(get_binary_data(i, num_variables))
    return tmp


def get_box_format(filenames, num_variables, boxsize=16):
    """
    Reads multiple binary files and reshapes the combined data into a specified 'box' format.

    The function reads data from files matching the given pattern, combines them into a tensor,
    and then reshapes this tensor to have a specified 'box' size along spatial dimensions.

    Parameters:
    filenames (str): A pattern string to match filenames. Can include wildcard characters.
    num_variables (int): The number of variables (doubles) to read from each file.
    boxsize (int): The size of the box to reshape each spatial dimension.

    Returns:
    torch.Tensor: A PyTorch tensor containing the reshaped data.
    """
    tmp = get_many_binary_files(filenames, num_variables)
    tmp = torch.tensor(tmp)
    sizevector = tmp.shape[-1]
    numberfiles = tmp.shape[0]
    # Reshape the tensor to the specified box size
    tmp = tmp.reshape(-1, boxsize, boxsize, boxsize, sizevector)
    # Making sure (0,1,2) = x,y,z corresponds to coords in GRChombo (reshape produces (0,1,2) = z,y,x)
    tmp = tmp.permute(0, 3, 2, 1, 4)
    return tmp


def cut_ghosts(tensor: torch.Tensor) -> torch.Tensor:
    """
    Trims 'ghost' cells from the spatial dimensions of a tensor.

    This function removes 2 cells from both ends of each of the second,
    third, and fourth dimensions (1, 2, 3 in zero-indexing), reducing
    each of these dimensions by a total of 4 cells.

    Parameters:
    tensor (torch.Tensor): The tensor to be trimmed.

    Returns:
    torch.Tensor: The trimmed tensor.
    """
    for dim in range(1, 4):
        tensor = tensor.narrow(dim, 2, tensor.size(dim) - 4)
    return tensor


# A list with all the keys (including derivative keys )
keys_all = [
    "chi",
    "h11",
    "h12",
    "h13",
    "h22",
    "h23",
    "h33",
    "K",
    "A11",
    "A12",
    "A13",
    "A22",
    "A23",
    "A33",
    "Theta",
    "Gamma1",
    "Gamma2",
    "Gamma3",
    "lapse",
    "shift1",
    "shift2",
    "shift3",
    "B1",
    "B2",
    "B3",
    "dx_chi",
    "dx_h11",
    "dx_h12",
    "dx_h13",
    "dx_h22",
    "dx_h23",
    "dx_h33",
    "dx_K",
    "dx_A11",
    "dx_A12",
    "dx_A13",
    "dx_A22",
    "dx_A23",
    "dx_A33",
    "dx_Theta",
    "dx_Gamma1",
    "dx_Gamma2",
    "dx_Gamma3",
    "dx_lapse",
    "dx_shift1",
    "dx_shift2",
    "dx_shift3",
    "dx_B1",
    "dx_B2",
    "dx_B3",
    "dy_chi",
    "dy_h11",
    "dy_h12",
    "dy_h13",
    "dy_h22",
    "dy_h23",
    "dy_h33",
    "dy_K",
    "dy_A11",
    "dy_A12",
    "dy_A13",
    "dy_A22",
    "dy_A23",
    "dy_A33",
    "dy_Theta",
    "dy_Gamma1",
    "dy_Gamma2",
    "dy_Gamma3",
    "dy_lapse",
    "dy_shift1",
    "dy_shift2",
    "dy_shift3",
    "dy_B1",
    "dy_B2",
    "dy_B3",
    "dz_chi",
    "dz_h11",
    "dz_h12",
    "dz_h13",
    "dz_h22",
    "dz_h23",
    "dz_h33",
    "dz_K",
    "dz_A11",
    "dz_A12",
    "dz_A13",
    "dz_A22",
    "dz_A23",
    "dz_A33",
    "dz_Theta",
    "dz_Gamma1",
    "dz_Gamma2",
    "dz_Gamma3",
    "dz_lapse",
    "dz_shift1",
    "dz_shift2",
    "dz_shift3",
    "dz_B1",
    "dz_B2",
    "dz_B3",
    "Ham",
    "Mom1",
    "Mom2",
    "Mom3",
]

keys = [
    "chi",
    "h11",
    "h12",
    "h13",
    "h22",
    "h23",
    "h33",
    "K",
    "A11",
    "A12",
    "A13",
    "A22",
    "A23",
    "A33",
    "Theta",
    "Gamma1",
    "Gamma2",
    "Gamma3",
    "lapse",
    "shift1",
    "shift2",
    "shift3",
    "B1",
    "B2",
    "B3",
]
