import numpy as np

def mode_n_product(tensor, matrix, mode):
    """
    Perform mode-n product of a tensor with a matrix.
    
    Parameters:
    tensor (ndarray): Input tensor of shape (I1, I2, ..., In, ..., IN)
    matrix (ndarray): Matrix to multiply with, of shape (J, In)
    mode (int): Mode along which to multiply (1-based index)
    
    Returns:
    ndarray: Resulting tensor after mode-n multiplication
    """
    # Convert to 0-based index for internal operations
    mode -= 1

    # Check if dimensions are compatible for multiplication
    if tensor.shape[mode] != matrix.shape[1]:
        raise ValueError(f"Cannot multiply: tensor's mode-{mode+1} dimension ({tensor.shape[mode]}) "
                         f"does not match matrix's second dimension ({matrix.shape[1]}).")
    
    # Move the mode-n dimension to the first dimension
    tensor = np.moveaxis(tensor, mode, 0)

    # Unfold the tensor into a matrix
    unfolded_tensor = tensor.reshape(tensor.shape[0], -1)

    # Perform matrix multiplication
    result = np.dot(matrix, unfolded_tensor)

    # Fold the resulting matrix back into a tensor
    new_shape = list(tensor.shape)
    new_shape[0] = matrix.shape[0]
    new_shape = tuple(new_shape)
    result_tensor = result.reshape(new_shape)
    
    # Move the dimensions back to the original order
    result_tensor = np.moveaxis(result_tensor, 0, mode)
    return result_tensor

# Example tensor (2 x 3 x 4)
tensor = np.random.rand(2, 3, 4)
print("Original Tensor:")
print(tensor)

# Example matrix (5 x 3) to multiply along mode-2 (1-based index)
matrix = np.random.rand(5, 3)
print("\nMatrix:")
print(matrix)

try:
    # Perform mode-2 (1-based index) multiplication
    result_tensor = mode_n_product(tensor, matrix, 2)
    print("\nResulting Tensor after mode-2 multiplication:")
    print(result_tensor)
except ValueError as e:
    print(f"\nError: {e}")

# Example of incompatible matrix (5 x 7)
incompatible_matrix = np.random.rand(5, 7)
print("\nIncompatible Matrix:")
print(incompatible_matrix)

try:
    # Attempt to perform mode-2 multiplication with an incompatible matrix
    result_tensor = mode_n_product(tensor, incompatible_matrix, 2)
    print("\nResulting Tensor after mode-2 multiplication with incompatible matrix:")
    print(result_tensor)
except ValueError as e:
    print(f"\nError: {e}")
