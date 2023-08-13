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

        min_rank = int(input("Please enter the minimum rank: "))
        max_rank = int(input("Please enter the maximum rank: "))

        print(
            f"Please rank the importance of each criteria on a scale from {min_rank}-{max_rank}, where {min_rank} indicates equal importance and {max_rank} indicates extreme importance.")

        criteria_importance = {}
        criteria_direction = {}
        for i, raster_file in enumerate(self.raster.keys()):
            importance = float(input(f"Please enter the importance for criteria '{raster_file}': "))
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
        print(f"Consistency ratio: {cr}")

        # Save the ahp weightings and direction as attributes
        self.criteria_weights = dict(zip(self.raster.keys(), weights))
        self.criteria_direction = criteria_direction

        # Print the final weights for each raster layer
        print("Final weights for each raster layer:")
        for raster_file, weight in self.criteria_weights.items():
            print(f"{raster_file}: {weight}")

    def fahp(self):
        """Get the weights and directions of each criteria using FAHP with pydecision."""

        # Number of criteria
        num_criteria = len(self.raster)

        # Prompt the user to define the scale
        scale = input("Please define the scale (e.g., 1-9): ")

        # Create an empty list to store the importance values for each criterion
        criteria_importance = []

        print("\nPlease provide three values representing the importance of each criterion:")

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

        print(f"\nConsistency Ratio (RC): {round(rc, 2)}")
        if rc > 0.10:
            print('The solution is inconsistent, the pairwise comparisons must be reviewed')
        else:
            print('The solution is consistent')

        # Save the FAHP weightings and direction as attributes
        self.criteria_weights = dict(zip(self.raster.keys(), normalized_weights))
        self.criteria_direction = criteria_direction

    def process_skcriteria(self):
        """Transform the data for skcriteria."""
        decision_matrix = []

        for raster_file, raster_path in self.raster.items():
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1).astype(float)

                # Debugging print
                print(f"Raster: {raster_file}, Min Value: {np.nanmin(raster_data)}, Max Value: {np.nanmax(raster_data)}")

                # Handle extreme negative values
                extreme_values = [-1.7976931348623157e+308, -3.4028234663852886e+38]
                for extreme in extreme_values:
                    raster_data[raster_data == extreme] = np.nan

                # Handle invalid values
                raster_data[np.isinf(raster_data)] = 0
                raster_data[np.isnan(raster_data)] = 1e-10  

                # Ensure no negative values remain
                raster_data[raster_data < 0] = 0

                decision_matrix.append(raster_data.reshape(-1))

        decision_matrix = np.array(decision_matrix).T
        objectives = [1 if self.criteria_direction[raster_file] else -1 for raster_file in self.raster.keys()]
        weights = [self.criteria_weights[raster_file] for raster_file in self.raster.keys()]

        return decision_matrix, weights, objectives

    
    def process_pydecision(self):
        """Transform the data for pyDecision."""

        # Create an empty list to store the 1D arrays
        decision_matrix = []

        for raster_file, raster_path in self.raster.items():
            # Open the raster file with rasterio
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1).astype(float)

            # Handle extreme values in raster data
            extreme_values = [-1.7976931348623157e+308, -3.4028234663852886e+38]
            for extreme in extreme_values:
                raster_data[raster_data == extreme] = np.nan

            # Debugging print to show the range of values for each raster
            print(f"Raster: {raster_file}, Min Value: {np.nanmin(raster_data)}, Max Value: {np.nanmax(raster_data)}")

            # Handle invalid values in raster data and
            # Replace NaN with a small value for computation
            raster_data[np.isinf(raster_data)] = 0
            raster_data[np.isnan(raster_data)] = 1e-10  

            decision_matrix.append(raster_data.reshape(-1))

        # Convert the list of 1D arrays to a 2D array
        decision_matrix = np.array(decision_matrix).T

        # Define the objectives (True for maximization and False for minimization)
        objectives = [self.criteria_direction[raster_file] for raster_file in self.raster.keys()]

        # Define the weights
        weights = [self.criteria_weights[raster_file] for raster_file in self.raster.keys()]

        # Define criteria names
        criteria_names = list(self.raster.keys())

        # Define alternatives names (assuming each row in the decision matrix is an alternative)
        alternatives_names = [f"Alternative_{i}" for i in range(1, decision_matrix.shape[0] + 1)]

        return decision_matrix, weights, criteria_names, alternatives_names, objectives

    def weighted_sum(self):
        """Calculate the suitability score using Weighted Sum with skcriteria."""
        decision_matrix, weights, objectives = self.process_skcriteria()
        weighted_sum = simple.WeightedSumModel()
        rank = weighted_sum.evaluate(DecisionMatrix(decision_matrix, weights=weights, objectives=objectives))
        weighted_score = rank.e_.score

        temp_raster = list(self.raster.values())[0]
        with rasterio.open(temp_raster) as src:
            weighted_score = weighted_score.reshape(src.shape)

            # Revert the small values back to NaN using the mask

            nan_mask = np.isnan(weighted_score)
            weighted_score[nan_mask] = np.nan

            with rasterio.open('weighted_sum_scores.tif', 'w', driver='GTiff', 
                            height=weighted_score.shape[0],
                            width=weighted_score.shape[1], 
                            count=1, dtype=str(weighted_score.dtype),
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(weighted_score, 1)

        return weighted_score

    def topsis(self):
        """Calculate the suitability score using TOPSIS."""

        # Use the process_skcriteria method to get the data structure needed for skcriteria package
        decision_matrix, weights, objectives = self.process_skcriteria()

        # Create a TOPSIS object
        topsis = TOPSIS(metric='euclidean')

        # Run TOPSIS
        decision = topsis.evaluate(DecisionMatrix(decision_matrix, objectives, weights))

        # Extract the similarity scores
        topsis_score = decision.e_.similarity

        # Reshape the 1D array back to a 2D array with the same shape as the raster data
        temp_raster_path = list(self.raster.values())[0] 
        with rasterio.open(temp_raster_path) as src:
            topsis_score = topsis_score.reshape(src.shape)  
            crs = src.crs
            transform = src.transform

        # Reassign NaN values to the final output raster
        topsis_score[np.isnan(topsis_score)] = np.nan

        # Write the topsis similarity scores to a GeoTIFF file
        with rasterio.open('topsis_scores.tif', 'w', driver='GTiff',
                    height=topsis_score.shape[0], width=topsis_score.shape[1],
                    count=1, dtype=topsis_score.dtype,
                    crs=crs, transform=transform) as dst:
            dst.write(topsis_score, 1)

        return topsis_score


project_directory = "hk_wind_turbine_site_selection_case_study"

gis_mcda = GisMcda("hk_wind_turbine_site_selection_case_study/criteria_layers",
                   "hk_wind_turbine_site_selection_case_study/study_area/study_area_without_constraints.shp")

gis_mcda.check_raster()
gis_mcda.transform_raster()

check_raster("hk_wind_turbine_site_selection_case_study/criteria_layers")
check_raster("hk_wind_turbine_site_selection_case_study/clipped_criteria_layers")

gis_mcda.ahp()
print(gis_mcda.criteria_weights)
print(gis_mcda.criteria_direction)
print("The following is weighted sum --------------------------------------------------------------------")
gis_mcda.weighted_sum()

print("The following is topsis --------------------------------------------------------------------")
gis_mcda.topsis()