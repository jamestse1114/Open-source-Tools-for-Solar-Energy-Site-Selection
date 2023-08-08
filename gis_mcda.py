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

directory_name = "hk_wind_turbine_site_selection_case_study"

def get_directory_name():
    # Ask the user for the directory name
    directory_name = input("Please enter the directory name: ")

    # Check if the directory exists
    if not os.path.exists(directory_name):
        print(f"The directory {directory_name} does not exist.")
        return None

    print(directory_name)
    return directory_name

def study_area_removed_contraints(directory_name, study_area_file_name, constraint_layers_folder_name):
    study_area_path = os.path.join(directory_name, 'data', study_area_file_name)
    constraint_layers_path = os.path.join(directory_name, constraint_layers_folder_name)
    
    # Load the study area shapefile
    study_area = gpd.read_file(study_area_path)

    # Start a list to contain the constraints
    constraints = []

    # Loop through the constraint shapefiles
    for constraint_file in os.listdir(constraint_layers_path):
        
        # Only load the shapefiles
        if constraint_file.endswith('.shp'):
            
            # Load the constraint shapefile
            constraint = gpd.read_file(os.path.join(constraint_layers_path, constraint_file))
            
            # Convert the crs of the constraint layers to the crs of the study area
            if constraint.crs != study_area.crs:
                constraint = constraint.to_crs(study_area.crs)
            
            # Add the constraints to the list
            constraints.append(constraint)

    # Concatenate all GeoDataFrames in the list
    constraints = pd.concat(constraints)

    # Perform the spatial operation to deduct the constraint area from the study area
    processed_study_area = gpd.overlay(study_area, constraints, how='difference')

    # Plot the study area which does not include constraint area
    processed_study_area.plot()
    plt.show()

    # Save the processed study area to a new shapefile
    output_path = os.path.join(directory_name, 'study_area', 'study_area_without_constraints.shp')
    processed_study_area.to_file(output_path)

def reclassify_raster_layer(directory_name, input_file_name, output_folder_name, reclass_dictionary):  
    input_file_path = os.path.join(directory_name, 'data', input_file_name)
    output_folder_path = os.path.join(directory_name, "criteria_layers")
    
    # Check if the input file exists
    if not os.path.exists(input_file_path):
        print(f"The file {input_file_path} does not exist.")
        return

    # Load the raster layer
    with rasterio.open(input_file_path) as layer:
        raster_data = layer.read(1)
        meta = layer.meta

    # Print the min and max values of the raster data
    print(f"Min value: {np.min(raster_data)}")
    print(f"Max value: {np.max(raster_data)}")

    # Print the class definitions
    print("Class definitions:")
    for class_num, class_value in reclass_dictionary.items():
        print(f"Class {class_num}: {class_value}")

    # Create a new array for the reclassified data
    reclassified = np.zeros(raster_data.shape, dtype=np.uint8)

    # Apply the reclassification
    for new_value, (low, high) in reclass_dictionary.items():
        reclassified[(raster_data >= low) & (raster_data < high)] = new_value

    # Create a new folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Write the reclassified data to a new raster file
    output_file_path = os.path.join(output_folder_path, f'reclassified_{os.path.basename(input_file_path)}')
    with rasterio.open(output_file_path, 'w', **meta) as dest:
        dest.write(reclassified, 1)

def plot_raster(directory_name, file_name):
    file_path = os.path.join(directory_name, file_name)

    # Open the raster file
    with rasterio.open(file_path) as src:
        # Read the raster data
        raster_data = src.read(1)

    # Create a new figure
    plt.figure(figsize=(10,10))

    # Display the raster data
    plt.imshow(raster_data, cmap='viridis')

    # Add a colorbar
    plt.colorbar(label='Raster Value')

    # Display the plot
    plt.show()

def plot_shapefile(directory_name, file_name, base_map_file_name=None):
    file_path = os.path.join(directory_name, file_name)

    # Load the shapefile
    gdf = gpd.read_file(file_path)

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10,10))

    # If a base map file name is provided, load and plot it first
    if base_map_file_name:
        base_map_path = os.path.join(directory_name, base_map_file_name)
        base_map = gpd.read_file(base_map_path)
        
        # Convert the CRS of the shapefile to match the base map
        gdf = gdf.to_crs(base_map.crs)
        
        base_map.plot(ax=ax, color='white', edgecolor='black')

    # Plot the shapefile
    gdf.plot(ax=ax, color='red')

    # Grid
    ax.grid(True)

    # North arrow
    arrow = FancyArrow(0.9, 0.85, dx=0, dy=0.05, width=0.01, color='k', transform=fig.transFigure)
    fig.patches.append(arrow)
    plt.text(0.9, 0.9, 'N', transform=fig.transFigure)

    plt.show()

def check_raster_shapes(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            with rasterio.open(os.path.join(directory, filename)) as dataset:
                array = dataset.read(1)
                print(array.shape)

class GisMcda: 
    def __init__(self, raster_directory, boundary_path):
        self.raster_directory = raster_directory
        self.boundary_path = boundary_path
        self.boundary = gpd.read_file(boundary_path)
        self.rasters = {}
        self.clipped_directory = os.path.join(os.path.dirname(raster_directory), 'clipped_criteria_layers')
        os.makedirs(self.clipped_directory, exist_ok=True)
        self.process_rasters()

    def process_rasters(self):
        min_res = float('inf')

        # find the largest resolution in the raster layers
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
                    transform, width, height = calculate_default_transform(src.crs, self.boundary.crs, src.width, src.height, *src.bounds, resolution=min_res)
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': self.boundary.crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })

                    reprojected = np.empty((src.count, height, width))
                    reproject(src.read(), reprojected, src_transform=src.transform, src_crs=src.crs,
                            dst_transform=transform, dst_crs=self.boundary.crs, resampling=Resampling.nearest)

                    out_image, out_transform = mask(src, [mapping(self.boundary.geometry[0])], crop=True)
                    out_meta = kwargs.copy()
                    out_meta.update({"driver": "GTiff",
                                    "height": out_image.shape[1],
                                    "width": out_image.shape[2],
                                    "transform": out_transform})

                    clipped_raster_path = os.path.join(self.clipped_directory, raster_file.replace(".tif", "_clipped.tif"))
                    try:
                        if os.path.exists(clipped_raster_path):
                            os.remove(clipped_raster_path)
                        with rasterio.open(clipped_raster_path, 'w', **out_meta) as dst:
                            dst.write(reprojected)
                    except Exception as e:
                        print(f"Could not open or delete the file {clipped_raster_path}. Error: {str(e)}")
                        continue

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


# study_area_removed_contraints(get_directory_name(), 'hk_boundary.shp', 'constraint_layers')

# reclassification_dict = {1: (150, float('inf')),
#     2: (120, 150),
#     3: (80, 120),
#     4: (40, 80),
#     5: (float('-inf'), 40)}
# reclassify_raster_layer(directory_name, 'elevation.tif', 'criteria_layers', reclassification_dict)

# plot_raster(directory_name, 'criteria_layers/reclassified_elevation.tif')

# plot_shapefile(directory_name, 'study_area/study_area_without_constraints.shp', 'data/hk_boundary.shp')

# print(check_raster_shapes("hk_wind_turbine_site_selection_case_study/clipped_criteria_layers"))

gis_mcda = GisMcda(os.path.join(directory_name, 'criteria_layers'), os.path.join(directory_name, 'study_area/study_area_without_constraints.shp'))
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
# gis_mcda.ahp()

g = gis_mcda.topsis()
print(gis_mcda.topsis())
print(g.shape)
print(np.nanmin(g))
print(np.nanmax(g))

check_raster_shapes("hk_wind_turbine_site_selection_case_study/clipped_criteria_layers")
check_raster_shapes("hk_wind_turbine_site_selection_case_study/criteria_layers")

def check_raster_shapes(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            with rasterio.open(os.path.join(directory, filename)) as dataset:
                array = dataset.read(1)
                print(array.shape)
