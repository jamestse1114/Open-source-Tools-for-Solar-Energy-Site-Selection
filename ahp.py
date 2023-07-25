import numpy as np
import rasterio

def load_rasters(raster_files):
    rasters = []
    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            raster = src.read(1)  # Read the first (and only) band
        rasters.append(raster)
    return rasters

# Assuming raster_files is a list of your raster file paths
raster_layers = load_rasters(raster_files)

# Now pass these loaded raster layers and the pairwise comparison matrix to the AHP function
result = ahp(pairwise_matrix, raster_layers)

def ahp(pairwise_matrix, raster_files):
    # Check that the pairwise comparison matrix is square
    assert pairwise_matrix.shape[0] == pairwise_matrix.shape[1], "Pairwise comparison matrix must be square"
    
    # Check that the number of raster files matches the size of the pairwise comparison matrix
    assert pairwise_matrix.shape[0] == len(raster_files), "Number of raster files must match size of pairwise comparison matrix"
    
    # Calculate the eigenvector of the pairwise comparison matrix
    eigenvalues, eigenvectors = np.linalg.eig(pairwise_matrix)
    
    # The weights of the criteria are given by the eigenvector corresponding to the maximum eigenvalue
    weights = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Normalize the weights so that they sum to 1
    weights = weights / np.sum(weights)
    
    # Initialize a NumPy array to hold the weighted sum of the raster layers
    weighted_sum = None
    
    # For each raster file and corresponding weight
    for raster_file, weight in zip(raster_files, weights):
        # Read the raster file
        with rasterio.open(raster_file) as src:
            raster = src.read(1)  # Read the first (and only) band
            
            # If the weighted sum array has not been initialized, do so now
            if weighted_sum is None:
                weighted_sum = np.zeros_like(raster)
            
            # Add the weighted raster to the weighted sum array
            weighted_sum += weight * raster
    
    return weighted_sum
