import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping
from scipy.stats import rankdata
from skcriteria.preprocessing import invert_objectives
from skcriteria.madm import simple, similarity
from skcriteria.madm.similarity import TOPSIS
from skcriteria import DecisionMatrix

project_directory = "hk_wind_turbine_site_selection_case_study"

def prompt_directory_name():
    
    """Ask the user for the directory name and check if it exists."""
    
    project_directory = input("Enter the directory name: ")
    print(project_directionory)
    
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

    # Convert the crs of the constraint layers to the crs of the study area and append the constraint layers into the array
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
    
    file_path = os.path.join(directory, file_name)
    shape = gpd.read_file(file_path)

    fig, ax = plt.subplots(figsize=(10,10))

    if base_map:
        base_map_path = os.path.join(directory, base_map)
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
    
    file_path = os.path.join(directory, file_name)
    with rasterio.open(file_path) as raster:
        data = raster.read(1)

    plt.imshow(data, cmap='viridis')
    plt.colorbar(label='Raster Value')
    plt.show()

def check_raster(directory):
    
    """Print the dimensions of the raster layers in a directory."""
    
    for file_name in os.listdir(directory):
        if file_name.endswith(".tif"):
            with rasterio.open(os.path.join(directory, file_name)) as rasters:
                print(rasters)
                print(rasters.shape)
                print(f"Min value: {np.min(rasters)}")
                print(f"Max value: {np.max(rasters)}")
       
class GisMcda: 
    def __init__(self, raster_directory, boundary_path):
        self.raster_directory = raster_directory
        self.boundary_path = boundary_path
        self.boundary = gpd.read_file(boundary_path)
        self.rasters = {}
        self.clipped_directory = os.path.join(os.path.dirname(raster_directory), 'clipped_criteria_layers')
        os.makedirs(self.clipped_directory, exist_ok=True)

        # Raster processing
        self.transform_rasters()
        self.clip_rasters()
        
    def transform_rasters(self):
        
        """Transform all raster layers to the same resolution."""
        
        min_res = float('inf')

        # Find the minimum resolution among all rasters layers
        for raster_file in os.listdir(self.raster_directory):
            if raster_file.endswith('.tif'):
                raster_path = os.path.join(self.raster_directory, raster_file)
                with rasterio.open(raster_path) as src:
                    res = max(src.res)
                    if res < min_res:
                        min_res = res

        # transforming all the raster layers
        for raster_file in os.listdir(self.raster_directory):
            
            if raster_file.endswith('.tif'):
                raster_path = os.path.join(self.raster_directory, raster_file)
                with rasterio.open(raster_path) as src:
                    
                    # Calculate the transformation parameters
                    transform, width, height = calculate_default_transform(
                        src.crs, self.boundary.crs, src.width, src.height, *src.bounds, resolution=min_res
                    )
                    
                    # Set up metadata for the reprojected raster
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': self.boundary.crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })

                    # Reproject the raster
                    reprojected = np.empty((src.count, height, width))
                    reproject(src.read(), reprojected, src_transform=src.transform, src_crs=src.crs,
                              dst_transform=transform, dst_crs=self.boundary.crs, resampling=Resampling.nearest)
                    self.rasters[raster_file] = reprojected
                    
    def clip_rasters(self):
        """Clip all raster layers based on the boundary."""
        for raster_file in os.listdir(self.raster_directory):
            if raster_file.endswith('.tif'):
                raster_path = os.path.join(self.raster_directory, raster_file)
                with rasterio.open(raster_path) as src:
                    out_image, out_transform = mask(src, [mapping(self.boundary.geometry[0])], crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform
                    })
                    clipped_raster_path = os.path.join(self.clipped_directory, raster_file.replace(".tif", "_clipped.tif"))
                    if os.path.exists(clipped_raster_path):
                        try:
                            os.remove(clipped_raster_path)
                        except PermissionError:
                            print(f"Warning: Could not delete {clipped_raster_path} as it's in use. Skipping...")
                            continue
                    with rasterio.open(clipped_raster_path, 'w', **out_meta) as dst:
                        dst.write(out_image)
                self.rasters[raster_file] = rasterio.open(clipped_raster_path)

    def ahp(self):
        min_rank = int(input("Please enter the minimum rank: "))
        max_rank = int(input("Please enter the maximum rank: "))

        print(f"Please rank the importance of each criteria on a scale from {min_rank}-{max_rank}, where {min_rank} indicates equal importance and {max_rank} indicates extreme importance.")
        criteria_importance = {}
        for i, raster_file in enumerate(self.rasters.keys()):
            importance = float(input(f"Please enter the importance for criteria '{raster_file}': "))
            criteria_importance[raster_file] = importance

        # Matrix Calculation
        num_criteria = len(self.rasters)
        matrix = np.zeros((num_criteria, num_criteria))
        for i, criteria_i in enumerate(self.rasters.keys()):
            for j, criteria_j in enumerate(self.rasters.keys()):
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
        ri = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59, 16: 1.6, 17: 1.61, 18: 1.62, 19: 1.63, 20: 1.64}
        cr = ci / ri[num_criteria]
        print(f"Consistency ratio: {cr}")

        # Save the ahp weightings as an attribute
        self.ahp_weights = dict(zip(self.rasters.keys(), weights))

        # Print the final weights for each raster layer
        print("Final weights for each raster layer:")
        for raster_file, weight in self.ahp_weights.items():
            print(f"{raster_file}: {weight}")

        # Calculate the suitability score for AHP and Weighted Sum Calculation and Export it as an AHP-Weighted Sum Suitability Score
        suitability_score_sum = None
        raster_transform = None

        for raster_file, raster in self.rasters.items():
            raster_data = raster.read(1)
            if suitability_score_sum is None:
                suitability_score_sum = np.zeros_like(raster_data)
            suitability_score_sum += self.ahp_weights[raster_file] * raster_data

            raster_transform = raster.transform

        study_area_mask = rasterize([(x.geometry, 1) for i, x in self.boundary.iterrows()], out_shape=suitability_score_sum.shape, transform=raster.transform, fill=0, all_touched=True, dtype=np.uint8)
        suitability_score_sum = suitability_score_sum * study_area_mask

        # Write the weighted sum scores to a new GeoTIFF file
        with rasterio.open('ahp_weighted_sum_scores.tif', 'w', driver='GTiff', height=suitability_score_sum.shape[0], width=suitability_score_sum.shape[1], count=1, dtype=str(suitability_score_sum.dtype), crs=next(iter(self.rasters.values())).crs, transform=next(iter(self.rasters.values())).transform) as dst:
            dst.write(suitability_score_sum, 1)

    def topsis(self):
        # Create an empty list to store the 1D arrays
        decision_matrix = []

        for raster_file, raster in self.rasters.items():
            # Read and reshape the raster data to a 1D array
            raster_data = raster.read(1).astype(float)
            
            # Handle extreme values in raster data
            extreme_values = [-1.7976931348623157e+308, -3.4028234663852886e+38]
            for extreme in extreme_values:
                raster_data[raster_data == extreme] = np.nan

            # Debugging print to show the range of values for each raster
            print(f"Raster: {raster_file}, Min Value: {np.nanmin(raster_data)}, Max Value: {np.nanmax(raster_data)}")

            # Handle invalid values in raster data
            raster_data[np.isinf(raster_data)] = 0
            raster_data[np.isnan(raster_data)] = 1e-10  # Replacing NaN with a small value for computation

            decision_matrix.append(raster_data.reshape(-1))
        
        # Convert the list of 1D arrays to a 2D array
        decision_matrix = np.array(decision_matrix).T

        print(f"Shape of decision_matrix: {decision_matrix.shape}")
        
        # Define the objectives, 1 for maximization and -1 for minimization
        objectives = [1 if self.criteria_direction[raster_file] else -1 for raster_file in self.rasters.keys()]

        # Define the weights
        weights = [self.ahp_weights[raster_file] for raster_file in self.rasters.keys()]

        # Create a DecisionMatrix object
        mtx = DecisionMatrix(decision_matrix, objectives, weights)

        # Create a TOPSIS object
        topsis = TOPSIS(metric='euclidean')

        # Run TOPSIS
        decision = topsis.evaluate(mtx)

        # Assign the similarity scores to the topsis_score variable
        topsis_score = decision.e_.similarity

        # Reshape the 1D array back to a 2D array with the same shape as the raster data
        topsis_score = topsis_score.reshape(raster_data.shape)

        # Reassign NaN values to the final output raster
        topsis_score[np.isnan(raster_data)] = np.nan

        # Reshape the 1D array back to a 2D array with the same shape as the raster data
        topsis_score = topsis_score.reshape(raster_data.shape)  # Use raster_data.shape instead of self.sample_raster_data.shape

        # Reassign NaN values to the final output raster
        topsis_score[np.isnan(raster_data)] = np.nan

        return topsis_score


# reclassification_dict = {1: (150, float('inf')),
#     2: (120, 150),
#     3: (80, 120),
#     4: (40, 80),
#     5: (float('-inf'), 40)}
# reclassify_raster_layer(directory_name, 'elevation.tif', 'criteria_layers', reclassification_dict)

gis_mcda = GisMcda("hk_wind_turbine_site_selection_case_study/criteria_layers", "hk_wind_turbine_site_selection_case_study/study_area/study_area_without_constraints.shp")
# gis_mcda.ahp_weights = {'reclassified_elevation.tif': 0.1190476190476191, 'reclassified_railway.tif': 0.07142857142857147, 'reclassified_river.tif': 0.07142857142857144, 'reclassified_road.tif': 0.047619047619047616, 'reclassified_roughness.tif': 0.14285714285714288, 'reclassified_settlements.tif': 0.16666666666666669, 'reclassified_slope.tif': 0.09523809523809523, 'reclassified_vegetation.tif': 0.07142857142857144, 'reclassified_wind_speed.tif': 0.21428571428571427}
# gis_mcda.criteria_direction = {
#             'reclassified_elevation.tif': True,  # True for maximization, False for minimization
#             'reclassified_railway.tif': True,
#             'reclassified_river.tif': True,
#             'reclassified_road.tif': True,
#             'reclassified_roughness.tif': True,
#             'reclassified_settlements.tif': True,
#             'reclassified_slope.tif': True,
#             'reclassified_vegetation.tif': True,
#             'reclassified_wind_speed.tif': True
#         }
gis_mcda.ahp_weights = {'reclassified_elevation.tif': 0.1190476190476191, 
                        'reclassified_roughness.tif': 0.14285714285714288, 
                        'reclassified_slope.tif': 0.09523809523809523, 
                        'reclassified_wind_speed.tif': 0.21428571428571427}

gis_mcda.criteria_direction = {
            'reclassified_elevation.tif': True,  # True for maximization, False for minimization
            'reclassified_roughness.tif': True,
            'reclassified_slope.tif': True,
            'reclassified_wind_speed.tif': True
        }

g = gis_mcda.topsis()
print(gis_mcda.topsis())
print(g.shape)
print(np.nanmin(g))
print(np.nanmax(g))

check_raster("hk_wind_turbine_site_selection_case_study/clipped_criteria_layers")
check_raster("hk_wind_turbine_site_selection_case_study/criteria_layers")