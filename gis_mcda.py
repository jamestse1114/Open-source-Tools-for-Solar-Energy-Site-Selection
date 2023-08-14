import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import fiona
import rasterio
from rasterio import features
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from shapely.geometry import mapping, shape
from scipy.stats import rankdata
from skcriteria import DecisionMatrix
from skcriteria.preprocessing import invert_objectives
from skcriteria.madm import simple, similarity
from skcriteria.madm import electre
from skcriteria.madm.similarity import TOPSIS
from pyDecision.algorithm import fuzzy_ahp_method
from pyDecision.algorithm import fuzzy_topsis_method
from pymcdm.methods import VIKOR


def prompt_directory_name():
    """Ask the user for the directory name and check if it exists."""

    project_directory = input("Enter the directory name: ")
    print(project_directory)

    if not os.path.exists(project_directory):
        print(f"Directory {project_directory} not found.")
        return None

    return project_directory


def remove_study_area_constraints(directory, study_area_file, constraint_folder):
    """Remove constraint areas from the study area shapefile and save the result."""

    # Joining the paths for the shapefiles
    study_area_path = os.path.join(directory, 'data', study_area_file)
    constraint_path = os.path.join(directory, constraint_folder)

    # Read the shapefiles with geopandas and Make a list to store them
    study_area = gpd.read_file(study_area_path)
    constraints_list = []

    # Convert the crs of the constraint layers to the crs of the study area and append
    # the constraint layers into the array
    for file in os.listdir(constraint_path):
        if file.endswith('.shp'):
            constraint = gpd.read_file(os.path.join(constraint_path, file))
            if constraint.crs != study_area.crs:
                constraint = constraint.to_crs(study_area.crs)
            constraints_list.append(constraint)

    # Concatenate and spatial difference the constraint layers from the study area
    all_constraints = pd.concat(constraints_list)
    result_study_area = gpd.overlay(study_area, all_constraints, how='difference')

    # Plot the study area with constraint areas removed
    result_study_area.plot()
    plt.show()

    # Save the final study area to a new shapefile in "study_area" folder in the directory
    output_path = os.path.join(directory, 'study_area', 'study_area_without_constraints.shp')
    result_study_area.to_file(output_path)


def reclassify_raster(directory, input_file, reclass_dict):
    """Reclassify a .TIF raster file based on an input dictionary and save the result."""

    # Joining the paths for the raster layers
    input_path = os.path.join(directory, 'data', input_file)
    output_folder = os.path.join(directory, "criteria_layers")

    if not os.path.exists(input_path):
        print(f"The file {input_path} does not exist.")
        return

    # Load the raster layer and information
    with rasterio.open(input_path) as raster:
        data = raster.read(1)
        meta = raster.meta

    # Print the min and max values of the original raster data
    print(f"Min value: {np.min(data)}")
    print(f"Max value: {np.max(data)}")

    # Print the class definitions in the reclassification dictionary
    print("Class definitions:")
    for class_num, class_value in reclass_dict.items():
        print(f"Class {class_num}: {class_value}")

    # Create a new array for the reclassified data and Calculate the reclassified values
    reclassified_data = np.zeros(data.shape, dtype=np.uint8)
    for new_value, (low, high) in reclass_dict.items():
        reclassified_data[(data >= low) & (data < high)] = new_value

    # Save the reclassified layers in the folder "criteria_layers"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, f'reclassified_{os.path.basename(input_path)}')
    with rasterio.open(output_path, 'w', **meta) as dest:
        dest.write(reclassified_data, 1)


def plot_shapefile(directory_name, file_name, base_map_file_name=None):
    """Display a shapefile with an optional base map."""

    file_path = os.path.join(directory_name, file_name)
    shape = gpd.read_file(file_path)

    fig, ax = plt.subplots(figsize=(10, 10))

    if base_map_file_name:
        base_map_path = os.path.join(directory_name, base_map_file_name)
        base = gpd.read_file(base_map_path)
        shape = shape.to_crs(base.crs)
        base.plot(ax=ax, color='white', edgecolor='black')

    shape.plot(ax=ax, color='red')
    ax.grid(True)

    arrow = FancyArrow(0.9, 0.85, dx=0, dy=0.05, width=0.01, color='k', transform=fig.transFigure)
    fig.patches.append(arrow)
    plt.text(0.9, 0.9, 'N', transform=fig.transFigure)
    plt.show()


def plot_raster(directory_name, file_name):
    """Display a raster file with the directory name and file name."""

    file_path = os.path.join(directory_name, file_name)
    with rasterio.open(file_path) as raster:
        data = raster.read(1)

    plt.imshow(data, cmap='viridis')
    plt.colorbar(label='Raster Value')
    plt.show()


def check_raster(directory):
    """Print the dimensions of the raster layers in a directory."""

    for file_name in os.listdir(directory):
        if file_name.endswith(".tif"):
            with rasterio.open(os.path.join(directory, file_name)) as raster:
                print(raster)
                print(raster.shape)
                data = raster.read(1)
                
                min_value = np.min(data)
                max_value = np.max(data)
                print(f"Range: {min_value} to {max_value}")
      

class GisMcda:
    def __init__(self, raster_directory, boundary_path):
        self.raster_directory = raster_directory
        self.boundary_path = boundary_path
        self.boundary = gpd.read_file(boundary_path)
        self.raster = {os.path.basename(file): os.path.join(self.raster_directory, file) for file in os.listdir(self.raster_directory) if file.endswith('.tif')}
        self.clipped_directory = os.path.join(os.path.dirname(raster_directory), 'clipped_criteria_layers')
        os.makedirs(self.clipped_directory, exist_ok=True)

    def check_raster(self):
        """Check if all raster have the same grid, resolution, and shape."""

        resolutions = []
        shapes = []
        transforms = []

        for raster_file in os.listdir(self.raster_directory):
            if raster_file.endswith('.tif'):
                raster_path = os.path.join(self.raster_directory, raster_file)
                with rasterio.open(raster_path) as src:
                    resolutions.append(src.res)
                    shapes.append((src.width, src.height))
                    transforms.append(src.transform)

                    data = src.read(1)

                    # print the range for debugging
                    min_value = np.min(data)
                    max_value = np.max(data)
                    print(f"Range of {raster_file}: {min_value} to {max_value}")
                    
        # Check if all resolutions are the same
        if len(set(resolutions)) != 1:
            return False

        # Check if all shapes are the same
        if len(set(shapes)) != 1:
            return False

        # Check if all transforms are the same (i.e., they align to the same grid)
        if len(set(transforms)) != 1:
            return False

        print(f"Resolution: {resolutions}")
        print(f"Shapes: {shapes}")
        print(f"Transforms: {transforms}")

        return True
    
    def transform_raster(self):
        """Transform all raster layers to the same resolution, boundary, and alignment."""

        # Step 1: Determine the properties of the reference raster
        first_raster_path = [os.path.join(self.raster_directory, f) for f in os.listdir(self.raster_directory) if f.endswith('.tif')][0]
        with rasterio.open(first_raster_path) as ref_src:
            ref_transform = ref_src.transform
            ref_width = ref_src.width
            ref_height = ref_src.height
            ref_nodata = ref_src.nodata

        # Extract the boundary geometry from the shapefile
        with fiona.open(self.boundary_path, "r") as shapefile:
            geoms = [feature["geometry"] for feature in shapefile]

        # Determine the largest resolution among all rasters
        max_res = 0
        all_raster_paths = [os.path.join(self.raster_directory, f) for f in os.listdir(self.raster_directory) if f.endswith('.tif')]
        for raster_path in all_raster_paths:
            with rasterio.open(raster_path) as src:
                res = max(src.res)
                if res > max_res:
                    max_res = res

        # Step 2: Reproject, align, and clip each raster to match the reference raster and the largest resolution
        for filepath in all_raster_paths:
            with rasterio.open(filepath) as src:
                # Reproject and align the raster to match the reference raster and the largest resolution
                reprojected_data = np.empty((src.count, int(ref_height * ref_src.res[1] / max_res), int(ref_width * ref_src.res[0] / max_res)), dtype=src.dtypes[0])
                reproject(
                    source=rasterio.band(src, 1),
                    destination=reprojected_data[0],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_src.crs,
                    resampling=Resampling.nearest)

                # Define the kwargs dictionary
                kwargs = {
                    'driver': 'GTiff',
                    'height': reprojected_data.shape[1],
                    'width': reprojected_data.shape[2],
                    'count': src.count,
                    'dtype': reprojected_data.dtype,
                    'crs': ref_src.crs,
                    'transform': ref_transform,
                    'nodata': ref_nodata
                }

                # Use MemoryFile for the mask operation
                with MemoryFile() as memfile:
                    with memfile.open(**kwargs) as mem_raster:
                        mem_raster.write(reprojected_data)

                        # Clip the reprojected data to the boundary shapefile
                        out_image, out_transform = mask(mem_raster, geoms, crop=True, nodata=ref_nodata)
                
                # Update the directory to "clipped_criteria_layers"
                if not os.path.exists(self.clipped_directory):
                    os.makedirs(self.clipped_directory, exist_ok=True)
                clipped_raster_path = os.path.join(self.clipped_directory, f"clipped_{os.path.basename(filepath)}")

                # Save the clipped raster data
                kwargs = src.meta.copy()
                kwargs.update({
                    'transform': out_transform,
                    'width': out_image.shape[2],
                    'height': out_image.shape[1],
                    'nodata': ref_nodata,
                    'res': (max_res, max_res)
                })
                
                with rasterio.open(clipped_raster_path, "w", **kwargs) as dest:
                    dest.write(out_image)

        self.raster_directory = self.clipped_directory

        # Update the self.raster dictionary to point to the clipped rasters
        self.raster = {os.path.basename(file): os.path.join(self.raster_directory, file) for file in os.listdir(self.raster_directory) if file.endswith('.tif')}

    def ahp(self):
        """Get the weights and directions of each criteria using AHP."""
        
        print("\nThe following is ahp -----------------------------------------------------------------------")
        print("Please provide the value representing the importance of each criterion:")
        criteria_importance = {}
        criteria_direction = {}
        
        for i, raster_file in enumerate(self.raster.keys()):
            importance = float(input(f"Importance value for '{raster_file}': "))
            criteria_importance[raster_file] = importance
            direction = input(f"Is '{raster_file}' a maximization criterion? (yes/no): ").strip().lower()
            criteria_direction[raster_file] = direction == 'yes'

        # Matrix Calculation
        num_criteria = len(self.raster)
        matrix = np.zeros((num_criteria, num_criteria))
        for i, criteria_i in enumerate(self.raster.keys()):
            for j, criteria_j in enumerate(self.raster.keys()):
                if i == j:
                    matrix[i, j] = 1
                elif i < j:
                    matrix[i, j] = criteria_importance[criteria_i] / criteria_importance[criteria_j]
                else:
                    matrix[i, j] = 1 / (criteria_importance[criteria_j] / criteria_importance[criteria_i])

        print("Pairwise comparison matrix:")
        print(matrix)

        # Consistency Ratio Calculation
        eigvals, eigvecs = np.linalg.eig(matrix)
        max_index = np.argmax(eigvals)
        weights = np.real(eigvecs[:, max_index])
        weights = weights / np.sum(weights)

        lambda_max = np.sum(weights * np.sum(matrix, axis=1))
        ci = (lambda_max - num_criteria) / (num_criteria - 1)
        ri = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48,
              13: 1.56, 14: 1.57, 15: 1.59, 16: 1.6, 17: 1.61, 18: 1.62, 19: 1.63, 20: 1.64}
        cr = ci / ri[num_criteria]

        print(f"\nConsistency Ratio (RC): {cr}")
        if cr > 0.10:
            print('The solution is inconsistent (RC > 0.10), the pairwise comparisons must be reviewed')
        else:
            print('The solution is consistent (RC <= 0.10)')
            
        # Save the ahp weightings and direction as attributes
        self.criteria_weights = dict(zip(self.raster.keys(), weights))
        self.criteria_direction = criteria_direction

        # Print the final weights for each raster layer
        print("\nFinal weights for each raster layer:")
        for raster_file, weight in self.criteria_weights.items():
            print(f"{raster_file}: {weight}")

    def fahp(self):
        """Get the weights and directions of each criteria using FAHP with pydecision."""

        # Number of criteria
        num_criteria = len(self.raster)

        # Create an empty list to store the importance values for each criterion
        criteria_importance = []
        
        print("\nThe following is fahp -----------------------------------------------------------------------")
        print("Please provide three values representing the importance of each criterion:")

        criteria_direction = {}
        for raster_file in self.raster.keys():
            values = input(f"Importance values for '{raster_file}' (e.g., 4,5.3,6): ").split(',')
            criteria_importance.append((float(values[0]), float(values[1]), float(values[2])))
            direction = input(f"Is '{raster_file}' a maximization criterion? (yes/no): ").strip().lower()
            criteria_direction[raster_file] = direction == 'yes'

        # Construct the fuzzy pairwise comparison matrix
        dataset = []
        for i in range(num_criteria):
            row = []
            for j in range(num_criteria):
                if i == j:
                    row.append((1, 1, 1))
                else:
                    row.append(criteria_importance[j])
            dataset.append(row)

        # Call Fuzzy AHP Function from pydecision
        fuzzy_weights, defuzzified_weights, normalized_weights, rc = fuzzy_ahp_method(dataset)
        
        # Print the results
        print("\nFuzzy Weights:")
        for i, raster_file in enumerate(self.raster.keys()):
            print(f"{raster_file}: {np.around(fuzzy_weights[i], 3)}")

        print("\nCrisp Weights:")
        for i, raster_file in enumerate(self.raster.keys()):
            print(f"{raster_file}: {round(defuzzified_weights[i], 3)}")

        print("\nNormalized Weights:")
        for i, raster_file in enumerate(self.raster.keys()):
            print(f"{raster_file}: {round(normalized_weights[i], 3)}")

        print(f"\nConsistency Ratio (RC): {rc}")
        if rc > 0.10:
            print('The solution is inconsistent (RC > 0.10), the pairwise comparisons must be reviewed')
        else:
            print('The solution is consistent (RC <= 0.10)')

        # Save the FAHP weightings and direction as attributes
        self.criteria_weights = dict(zip(self.raster.keys(), normalized_weights))
        self.fuzzy_weights = dict(zip(self.raster.keys(), fuzzy_weights))
        self.criteria_direction = criteria_direction

    def process_skcriteria(self):
        """Transform the data for skcriteria."""
        decision_matrix = []
        valid_indices_list = []

        for raster_file, raster_path in self.raster.items():
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1).astype(float)

                # Debugging print
                print(f"Raster: {raster_file}, Min Value: {np.nanmin(raster_data)}, Max Value: {np.nanmax(raster_data)}")

                # Handle extreme negative values
                extreme_values = [-1.7976931348623157e+308, -3.4028234663852886e+38]
                for extreme in extreme_values:
                    raster_data[raster_data == extreme] = np.nan

                # Handle invalid values in raster data
                raster_data[np.isinf(raster_data)] = np.nan

                flattened_data = raster_data.reshape(-1)
                decision_matrix.append(flattened_data)

                valid_indices = np.where(~np.isnan(flattened_data))[0]
                valid_indices_list.append(set(valid_indices))

        # Ensure all rasters have the same valid indices
        common_valid_indices = list(set.intersection(*valid_indices_list))

        # Filter the decision matrix to only include common valid values
        decision_matrix = np.array(decision_matrix)[:, common_valid_indices].T

        objectives = [1 if self.criteria_direction[raster_file] else -1 for raster_file in self.raster.keys()]
        weights = [self.criteria_weights[raster_file] for raster_file in self.raster.keys()]

        return decision_matrix, weights, objectives, common_valid_indices

    def process_pymcdm(self):
        """Transform the data for pyDecision."""

        # Create an empty list to store the 1D arrays
        decision_matrix = []
        valid_indices_list = [] 
        
        for raster_file, raster_path in self.raster.items():
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1).astype(float)

                # Debugging print
                print(f"Raster: {raster_file}, Min Value: {np.nanmin(raster_data)}, Max Value: {np.nanmax(raster_data)}")

                # Handle extreme negative values
                extreme_values = [-1.7976931348623157e+308, -3.4028234663852886e+38]
                for extreme in extreme_values:
                    raster_data[raster_data == extreme] = np.nan

                # Handle invalid values in raster data
                raster_data[np.isinf(raster_data)] = np.nan

                flattened_data = raster_data.reshape(-1)
                decision_matrix.append(flattened_data)

                valid_indices = np.where(~np.isnan(flattened_data))[0]
                valid_indices_list.append(set(valid_indices))

        # Ensure all rasters have the same valid indices
        common_valid_indices = list(set.intersection(*valid_indices_list))

        # Filter the decision matrix to only include common valid values
        decision_matrix = np.array(decision_matrix)[:, common_valid_indices].T

        # Define the objectives (True for maximization and False for minimization)
        objectives = [self.criteria_direction[raster_file] for raster_file in self.raster.keys()]
        objectives = ['max' if obj else 'min' for obj in objectives]
        
        # Define the weights
        weights = [self.criteria_weights[raster_file] for raster_file in self.raster.keys()]
        
        # Define criteria names
        criteria_names = list(self.raster.keys())

        # Define alternatives names (assuming each row in the decision matrix is an alternative)
        alternatives_names = [f"Alternative_{i}" for i in range(1, decision_matrix.shape[0] + 1)]

        return decision_matrix, weights, criteria_names, alternatives_names, objectives, common_valid_indices
    
    def process_fuzzy(self):
        """Transform the data for pyDecision."""

        # Create an empty list to store the 1D arrays
        decision_matrix = []
        valid_indices_list = [] 
        
        for raster_file, raster_path in self.raster.items():
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1).astype(float)

                # Debugging print
                print(f"Raster: {raster_file}, Min Value: {np.nanmin(raster_data)}, Max Value: {np.nanmax(raster_data)}")

                # Handle extreme negative values
                extreme_values = [-1.7976931348623157e+308, -3.4028234663852886e+38]
                for extreme in extreme_values:
                    raster_data[raster_data == extreme] = np.nan

                # Handle invalid values in raster data
                raster_data[np.isinf(raster_data)] = np.nan

                flattened_data = raster_data.reshape(-1)
                decision_matrix.append(flattened_data)

                valid_indices = np.where(~np.isnan(flattened_data))[0]
                valid_indices_list.append(set(valid_indices))

        # Ensure all rasters have the same valid indices
        common_valid_indices = list(set.intersection(*valid_indices_list))

        # Filter the decision matrix to only include common valid values
        decision_matrix = np.array(decision_matrix)[:, common_valid_indices].T

        # Define the objectives (True for maximization and False for minimization)
        objectives = [self.criteria_direction[raster_file] for raster_file in self.raster.keys()]
        objectives = ['max' if obj else 'min' for obj in objectives]
        
        # Define the weights
        weights = [self.fuzzy_weights[raster_file] for raster_file in self.raster.keys()]
        
        # Define criteria names
        criteria_names = list(self.raster.keys())

        # Define alternatives names (assuming each row in the decision matrix is an alternative)
        alternatives_names = [f"Alternative_{i}" for i in range(1, decision_matrix.shape[0] + 1)]

        return decision_matrix, weights, criteria_names, alternatives_names, objectives, common_valid_indices

    def saw(self):
        """Calculate the suitability score using Simple Additive Weighting with skcriteria."""
        
        print("\nThe following is Simple Additive Weighting --------------------------------------------------------------")
        
        decision_matrix, weights, objectives, valid_indices = self.process_skcriteria()
        weighted_sum = simple.WeightedSumModel()
        rank = weighted_sum.evaluate(DecisionMatrix(decision_matrix, weights=weights, objectives=objectives))
        weighted_score = rank.e_.score

        # Create an empty array filled with NaN values with the shape of the original raster
        temp_raster = list(self.raster.values())[0]
        with rasterio.open(temp_raster) as src:
            output_array = np.full(src.shape, np.nan)
            
            # Assign the computed scores to their respective positions in the output array
            np.put(output_array, valid_indices, weighted_score)
            
            # Write the output array to a new raster file
            with rasterio.open('weighted_sum_scores.tif', 'w', driver='GTiff', 
                            height=output_array.shape[0],
                            width=output_array.shape[1], 
                            count=1, dtype=str(output_array.dtype),
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(output_array, 1)

        return output_array

    def topsis(self):
        """Calculate the suitability score using TOPSIS."""

        print("\nThe following is TOPSIS --------------------------------------------------------------------")
        
        # Use the process_skcriteria method to get the data structure needed for skcriteria package
        decision_matrix, weights, objectives, valid_indices = self.process_skcriteria()

        # Create a TOPSIS object
        topsis = TOPSIS(metric='euclidean')

        # Run TOPSIS
        decision = topsis.evaluate(DecisionMatrix(decision_matrix, objectives, weights))

        # Extract the similarity scores
        topsis_score = decision.e_.similarity

        temp_raster_path = list(self.raster.values())[0] 
        with rasterio.open(temp_raster_path) as src:
            # Initialize an array filled with NaN values
            full_raster_scores = np.full(src.shape, np.nan)
            
            # Map the TOPSIS scores back to the original raster shape using the valid indices
            full_raster_scores.ravel()[valid_indices] = topsis_score

            with rasterio.open('topsis_scores.tif', 'w', driver='GTiff',
                            height=full_raster_scores.shape[0], width=full_raster_scores.shape[1],
                            count=1, dtype=full_raster_scores.dtype,
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(full_raster_scores, 1)

        return full_raster_scores
    
    def ftopsis(self, delta = 0.1, graph=False, verbose=False):
        """Calculate the suitability score using Fuzzy TOPSIS."""
        
        print("\nThe following is Fuzzy TOPSIS --------------------------------------------------------------------")
        
        # Use the fuzzy method to get the data structure needed
        decision_matrix, weights, criteria_names, alternatives_names, objectives, valid_indices = self.process_fuzzy()

        # Convert the decision_matrix to fuzzy numbers
        fuzzy_decision_matrix = []
        for row in decision_matrix:
            fuzzy_row = [(value-delta, value, value+delta) for value in row]
            fuzzy_decision_matrix.append(fuzzy_row)

        # Convert objectives to the format expected by pyDecision
        criterion_type = ['max' if obj == 'max' else 'min' for obj in objectives]
            
        # Use the fuzzy_topsis method from pyDecision
        scores = fuzzy_topsis_method(fuzzy_decision_matrix, [weights], criterion_type, graph, verbose)
            
        # Map the scores back to the original raster shape
        temp_raster_path = list(self.raster.values())[0]
        with rasterio.open(temp_raster_path) as src:
            full_raster_scores = np.full(src.shape, np.nan)
                
            # This is the crucial step: mapping the computed scores back to their original positions
            full_raster_scores.ravel()[valid_indices] = scores

            # Validation for mapping
            assert not np.isnan(full_raster_scores).all(), "All values in full_raster_scores are NaN!"

            with rasterio.open("fuzzy_topsis_scores.tif", 'w', driver='GTiff',
                            height=full_raster_scores.shape[0], width=full_raster_scores.shape[1],
                            count=1, dtype=full_raster_scores.dtype,
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(full_raster_scores, 1)
                    
        return full_raster_scores
    
    def vikor(self):
        print("\nThe following is VIKOR --------------------------------------------------------------------")

        # Use the process_pymcdm method to get the data structure needed
        decision_matrix, weights, criteria_names, alternatives_names, objectives, valid_indices = self.process_pymcdm()

        # Convert objectives to 1 (benefit) or -1 (cost) format for VIKOR in pymcdm
        criteria_types = np.array([1 if obj == 'max' else -1 for obj in objectives])

        # Ensure that decision_matrix, weights, and criteria_types are numpy arrays
        decision_matrix = np.array(decision_matrix)
        weights = np.array(weights)
        criteria_types = np.array(criteria_types)

        # Compute VIKOR scores using pymcdm
        vikor_body = VIKOR()
        q_values = vikor_body(decision_matrix, weights, criteria_types)

        # Normalize and reverse the suitability scores
        min_q = min(q_values)
        max_q = max(q_values)
        q_values = [(max_q - q) / (max_q - min_q) for q in q_values]
        
        # Map the scores back to the original raster shape
        temp_raster_path = list(self.raster.values())[0]
        with rasterio.open(temp_raster_path) as src:
            full_raster_image = np.full(src.shape, np.nan)
            
            # This is the crucial step: mapping the computed scores back to their original positions
            full_raster_image.ravel()[valid_indices] = q_values

            # Validation for mapping
            assert not np.isnan(full_raster_image).all(), "All values in full_raster_image are NaN!"

            with rasterio.open("vikor_scores.tif", 'w', driver='GTiff',
                            height=full_raster_image.shape[0], width=full_raster_image.shape[1],
                            count=1, dtype=full_raster_image.dtype,
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(full_raster_image, 1)

        return q_values


if __name__ == "__main__":
    
    project_directory = "hk_wind_turbine_site_selection_case_study"

    gis_mcda = GisMcda("hk_wind_turbine_site_selection_case_study/criteria_layers",
                    "hk_wind_turbine_site_selection_case_study/study_area/study_area_without_constraints.shp")

    gis_mcda.check_raster()
    gis_mcda.transform_raster()

    print("\nThe following are criteria_layers -----------------------------------------------------------------------")
    check_raster("hk_wind_turbine_site_selection_case_study/criteria_layers")

    print("\nThe following are clipped_criteria_layers -----------------------------------------------------------------------")
    check_raster("hk_wind_turbine_site_selection_case_study/clipped_criteria_layers")

    gis_mcda.fahp()
    gis_mcda.saw()
    gis_mcda.topsis()
    gis_mcda.vikor()
    gis_mcda.ftopsis()