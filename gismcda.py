import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import matplotlib.colors as mcolors
import seaborn as sns
import mplleaflet
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
from skcriteria.madm import simple, similarity
from skcriteria.madm.similarity import TOPSIS
from pyDecision.algorithm import fuzzy_ahp_method, ranking
from pymcdm.methods import VIKOR
from pyfdm import methods
from pyds import MassFunction


def remove_study_area_constraints(directory, study_area, constraints):
    """
        Remove constraint areas from the study area shapefile and 
        export the clipped shapefile.
    """

    # Read the shapefiles with geopandas and Make a list to store them
    study_area = gpd.read_file(study_area)
    constraints_list = []

    # Convert the crs of the constraint layers to the crs of the study area and 
    # append the constraint layers into the array
    for file in os.listdir(constraints):
        if file.endswith('.shp'):
            constraint = gpd.read_file(os.path.join(constraints, file))
            if constraint.crs != study_area.crs:
                constraint = constraint.to_crs(study_area.crs)
            constraints_list.append(constraint)

    # Spatial difference the constraint layers from the study area
    all_constraints = pd.concat(constraints_list)
    result_study_area = gpd.overlay(study_area, all_constraints, how='difference')

    # Plot the study area with constraint areas removed
    result_study_area.plot()
    plt.show()

    # Save the final study area to a new shapefile in "study_area" folder in the directory
    study_area_dir = os.path.join(directory, 'study_area')
    if not os.path.exists(study_area_dir):
        os.makedirs(study_area_dir)

    output_path = os.path.join(study_area_dir, 'study_area_without_constraints.shp')
    result_study_area.to_file(output_path)


def reclassify_raster(directory, raster, reclassification_dictionary):
    """
        Reclassify a .TIF raster based on an input dictionary and save the result.
    """

    criteria_layer_folder = os.path.join(directory, "criteria_layers")

    if os.path.exists(raster) != True:
        print(f"{raster} does not exist.")
        return

    # Load the raster layer and information
    with rasterio.open(raster) as rast:
        data = rast.read(1)
        meta = rast.meta

    # Print the min and max values of the original raster data
    print(f"Min value: {np.min(data)}")
    print(f"Max value: {np.max(data)}")

    # Print the class definitions in the reclassification dictionary
    print("Class definitions:")
    for class_num, class_value in reclassification_dictionary.items():
        print(f"Class {class_num}: {class_value}")

    # Create a new array for the reclassified data and Calculate the reclassified values
    reclassified_data = np.zeros(data.shape, dtype=np.uint8)
    for new_value, (low, high) in reclassification_dictionary.items():
        reclassified_data[(data >= low) & (data < high)] = new_value

    # Save the reclassified layers in the criteria_layers folder
    if os.path.exists(criteria_layer_folder) != True:
        os.makedirs(criteria_layer_folder)

    output_path = os.path.join(criteria_layer_folder, f'reclassified_{os.path.basename(raster)}')
    with rasterio.open(output_path, 'w', **meta) as rest:
        rest.write(reclassified_data, 1)


def check_raster(directory):
    """
        Check the dimensions and ranges of the raster layers in a directory.
    """

    for files in os.listdir(directory):
        if files.endswith(".tif"):
            with rasterio.open(os.path.join(directory, files)) as raster:
                print(raster)
                print(raster.shape)
                data = raster.read(1)
                
                min_value = np.min(data)
                max_value = np.max(data)
                print(f"Range: {min_value} to {max_value}")
                
      
def plot_raster(raster):
    """
        Display a raster file.
    """

    with rasterio.open(raster) as rast:
        data = rast.read(1)

    plt.imshow(data, cmap='viridis')
    plt.colorbar(label='Raster Value')
    plt.show()

          
def plot_shapefile(shapefile, base_map=None):
    """
        Display a shapefile with an optional base map.
    """

    shape = gpd.read_file(shapefile)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if base_map:
        base_map = gpd.read_file(base_map)
        shape = shape.to_crs(base_map.crs)
        base_map.plot(ax=ax, color='white', edgecolor='black')

    shape.plot(ax=ax, color='red')
    ax.grid(True)
    
    arrow = FancyArrow(0.9, 0.85, dx=0, dy=0.05, width=0.01, color='k', transform=fig.transFigure)
    fig.patches.append(arrow)
    plt.text(0.9, 0.9, 'N', transform=fig.transFigure)
    mplleaflet.show(fig=ax.figure)


class GisMcda:
    def __init__(self, raster_folder_path, boundary_path):
        self.raster_directory = raster_folder_path
        self.boundary_path = boundary_path
        self.boundary = gpd.read_file(boundary_path)
        self.raster = {os.path.basename(file):
            os.path.join(self.raster_directory, file)
            for file in os.listdir(self.raster_directory)
            if file.endswith('.tif')}
        self.clipped_directory = os.path.join(os.path.dirname(raster_folder_path), 'clipped_criteria_layers')
        os.makedirs(self.clipped_directory, exist_ok=True)

    def check_raster(self):
        """
            Check if all raster have the same alignment, grid, resolution, and shape.
        """

        resolutions = []
        shapes = []
        transforms = []

        for raster_file in os.listdir(self.raster_directory):
            if raster_file.endswith('.tif'):
                raster_path = os.path.join(self.raster_directory, raster_file)
                
                # get the value of res, width and height, transform for each raster
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

        # Check if raster align
        if len(set(transforms)) != 1:
            return False

        print(f"Resolution: {resolutions}")
        print(f"Shapes: {shapes}")
        print(f"Transforms: {transforms}")

        return True
    
    def transform_raster(self):
        """
            Transform all raster layers to the same resolution, boundary, and alignment.
        """

        # Get the properties of the reference raster
        first_raster_path = [os.path.join(self.raster_directory, f) 
                             for f in os.listdir(self.raster_directory) 
                             if f.endswith('.tif')][0]
        
        with rasterio.open(first_raster_path) as ref_src:
            ref_transform = ref_src.transform
            ref_width = ref_src.width
            ref_height = ref_src.height
            ref_nodata = ref_src.nodata

        # Get the boundary
        with fiona.open(self.boundary_path, "r") as shapefile:
            geoms = [feature["geometry"] for feature in shapefile]

        # Get the largest resolution in raster
        max_res = 0
        all_raster_path = [os.path.join(self.raster_directory, f) 
                            for f in os.listdir(self.raster_directory) 
                            if f.endswith('.tif')]
        for raster_path in all_raster_path:
            with rasterio.open(raster_path) as src:
                res = max(src.res)
                if res > max_res:
                    max_res = res
        
        # loop through each raster
        for filepath in all_raster_path:
            
            # Reproject and align to match the reference raster and the largest resolution
            with rasterio.open(filepath) as src:
                
                # Set the empty boundary for the reprojected raster first
                reprojected_data = np.empty((src.count, int(ref_height * ref_src.res[1] / max_res), 
                                             int(ref_width * ref_src.res[0] / max_res)), 
                                            dtype=src.dtypes[0])
                
                reproject(
                    source=rasterio.band(src, 1),
                    destination=reprojected_data[0],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_src.crs,
                    resampling=Resampling.nearest)

                # Define the info dictionary - reference crs and reprojection shape
                info = {
                    'driver': 'GTiff',
                    'height': reprojected_data.shape[1],
                    'width': reprojected_data.shape[2],
                    'count': src.count,
                    'dtype': reprojected_data.dtype,
                    'crs': ref_src.crs,
                    'transform': ref_transform,
                    'nodata': ref_nodata
                }

                # Save the masking info to memory file in rasterio
                with MemoryFile() as memoryfile:
                    with memoryfile.open(**info) as memoryraster:
                        memoryraster.write(reprojected_data)

                        # Save the transform shape for the meta update
                        out_image, out_transform = mask(memoryraster, geoms, crop=True, nodata=ref_nodata)
                
                if not os.path.exists(self.clipped_directory):
                    os.makedirs(self.clipped_directory, exist_ok=True)
                clipped_raster_path = os.path.join(self.clipped_directory, f"clipped_{os.path.basename(filepath)}")

                # Save the clipped raster data
                info = src.meta.copy()
                info.update({
                    'transform': out_transform,
                    'width': out_image.shape[2],
                    'height': out_image.shape[1],
                    'nodata': ref_nodata,
                    'res': (max_res, max_res)
                })
                
                with rasterio.open(clipped_raster_path, "w", **info) as rast:
                    rast.write(out_image)

        # Update the raster directory to the clipped directory
        self.raster_directory = self.clipped_directory

        # Update the self.raster dictionary for clipped directory raster
        self.raster = {os.path.basename(file): os.path.join(self.raster_directory, file) for file in os.listdir(self.raster_directory) if file.endswith('.tif')}

    def ahp(self):
        """
            Get the weight and direction of each criteria using AHP.
        """
        
        print("\nThe following is ahp -----------------------------------------------------------------------")
        print("Please provide the value representing the importance of each criterion:")
        criteria_importance = {}
        criteria_direction = {}
        
        # Save the criteria importance and criteria direction from user's input
        for k, raster_file in enumerate(self.raster.keys()):
            importance = float(input(f"Importance value for '{raster_file}': "))
            criteria_importance[raster_file] = importance
            direction = input(f"Is '{raster_file}' a maximization criterion? (yes/no): ").strip().lower()
            criteria_direction[raster_file] = direction == 'yes'

        # Computing a pairwise comparison matrix
        criteria_num = len(self.raster)
        matrix = np.zeros((criteria_num, criteria_num))
        
        # Looping and Comparing between criteria
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

        # Consistency Ratio (RC)
        eigenvalues, eigenvector = np.linalg.eig(matrix)
        max_index = np.argmax(eigenvalues)
        weights = np.real(eigenvector[:, max_index])
        weights = weights / np.sum(weights)

        max_eginv = np.sum(weights * np.sum(matrix, axis=1))
        ci = (max_eginv - criteria_num) / (criteria_num - 1)
        ri = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48,
              13: 1.56, 14: 1.57, 15: 1.59, 16: 1.6, 17: 1.61, 18: 1.62}
        cr = ci / ri[criteria_num]

        print(f"\nConsistency Ratio (RC): {cr}")
        if cr > 0.10:
            print('The solution is inconsistent (RC > 0.10), the pairwise comparisons must be reviewed')
        else:
            print('The solution is consistent (RC <= 0.10)')
        
        # Save weighting and direction as class attributes
        self.criteria_weights = dict(zip(self.raster.keys(), weights))
        self.criteria_direction = criteria_direction

        print(self.criteria_weights)
        print(self.criteria_direction)
        
        return self.criteria_direction, self.criteria_weights

    def fahp(self):
        """
            Get the fuzzy weight and direction of each criteria using FAHP by pydecision.
            It allows the user to handle uncertainty in the weighting process.
            The fuzzy method used is TFN.
            Note that both weight and fuzzy weight are returned.
        """

        criteria_importance = []
        criteria_direction = {}
        criteria_num = len(self.raster)
        
        print("\nThe following is fahp -----------------------------------------------------------------------")
        print("Please provide three values representing the importance of each criterion:")

        criteria_importance_dict = {}
        for raster_file in self.raster.keys():
            fuzzy_values = input(f"Importance values for '{raster_file}' (e.g., 4,5.3,6): ").split(',')
            criteria_importance_dict[raster_file] = (float(fuzzy_values[0]), float(fuzzy_values[1]), float(fuzzy_values[2]))
            direction = input(f"Is '{raster_file}' a maximization criterion? (yes/no): ").strip().lower()
            criteria_direction[raster_file] = direction == 'yes'

        # Compute a fuzzy pairwise comparison matrix
        fuzzy_matrix = []
        for raster_file_i in self.raster.keys():
            row = []
            for raster_file_j in self.raster.keys():
                if raster_file_i == raster_file_j:
                    row.append((1, 1, 1))
                else:
                    a, m, b = criteria_importance_dict[raster_file_i]
                    if a > 1:
                        row.append((1/b, 1/m, 1/a))
                    else:
                        row.append((a, m, b))
            fuzzy_matrix.append(row)

        # Fuzzy AHP Function from pydecision to get the fuzzy weights, defuzzified weights and normalised weights
        fuzzy_weights, defuzzified_weights, normalized_weights, rc = fuzzy_ahp_method(fuzzy_matrix)
        
        print("\nFuzzy Weights:")
        for k, raster_file in enumerate(self.raster.keys()):
            print(f"{raster_file}: {np.around(fuzzy_weights[k], 3)}")

        print("\nCrisp Weights:")
        for k, raster_file in enumerate(self.raster.keys()):
            print(f"{raster_file}: {round(defuzzified_weights[k], 3)}")

        print("\nNormalized Weights:")
        for k, raster_file in enumerate(self.raster.keys()):
            print(f"{raster_file}: {round(normalized_weights[k], 3)}")

        print(f"\nConsistency Ratio (RC): {rc}")
        if rc > 0.10:
            print('The solution is inconsistent (RC > 0.10), the pairwise comparisons must be reviewed')
        else:
            print('The solution is consistent (RC <= 0.10)')
        
        self.criteria_direction = criteria_direction
        self.criteria_weights = dict(zip(self.raster.keys(), normalized_weights))
        self.range_weights = dict(zip(self.raster.keys(), fuzzy_weights))
        
        return self.criteria_direction, self.criteria_weights, self.range_weights

    def dst(self, criteria_weights_list, uncertainty_factor=None):
        """
        Using Dempster-Shafer Theory to input a list of experts' weighting as evidences, and use 
        Bayesian theory to process and output a confidence interval of the weight of criteria. If 
        no uncertainty factor is provide, it will be assumed as 0.05.
        """
    
        if uncertainty_factor is None:
            uncertainty_factor = 0.05

        mass_functions = []

        # Process each expert's opinion
        for expert_weights in criteria_weights_list:
            mf = MassFunction()
            
            # Calculate the total mass based on the expert's weights
            total_mass = 0
            for criterion, weight in expert_weights.items():
                uncertainty_factor = max(0, min(1, uncertainty_factor))
                adjusted_weight = weight * (1 - uncertainty_factor)
                mf[frozenset([criterion])] = adjusted_weight
                total_mass += adjusted_weight
            
            # Assign the remaining mass to represent the expert's uncertainty
            mf[frozenset(expert_weights.keys())] = 1 - total_mass
            mass_functions.append(mf)

        # Combine the mass functions from all experts' opinions
        combined_mf = mass_functions[0]
        for next_mf in mass_functions[1:]:
            combined_mf = combined_mf & next_mf

        # Get belief and plausibility values (lower and upper)
        beliefs = combined_mf.bel()
        plausibilities = combined_mf.pl()

        # Save them to range weights and criteria weights
        range_weights = {}
        criteria_weights = {}
        for criteria_set, belief_value in beliefs.items():
            # Only process the weights for single criteria
            if len(criteria_set) == 1:
                the_criterion = list(criteria_set)[0]
                lower_bound = belief_value
                upper_bound = plausibilities[criteria_set]
                mean_weight = (lower_bound + upper_bound) / 2

                range_weights[the_criterion] = (lower_bound, mean_weight, upper_bound)
                criteria_weights[the_criterion] = mean_weight
        
        self.criteria_weights = criteria_weights
        self.range_weights = range_weights
        
        print("\nRange Weights:")
        print(self.range_weights)
        
        print("\nCriteria Weights:")
        print(self.criteria_weights)
        
        return self.range_weights, self.criteria_weights

    def process_skcriteria(self):
        """
            Transform the data for skcriteria package.
        """
        
        decision_matrix = []
        valid_indices_list = []

        for raster_file, raster_path in self.raster.items():
            with rasterio.open(raster_path) as src:
                rast = src.read(1).astype(float)

                # Debugging print
                print(f"Raster: {raster_file}, Min Value: {np.nanmin(rast)}, Max Value: {np.nanmax(rast)}")

                # Change extreme and invalid values to nan
                extreme_values = [-1.7976931348623157e+308, -3.4028234663852886e+38]
                for extreme in extreme_values:
                    rast[rast == extreme] = np.nan
                rast[np.isinf(rast)] = np.nan

                # Get the index for valid cells
                flattened_data = rast.reshape(-1)
                decision_matrix.append(flattened_data)

                valid_indices = np.where(~np.isnan(flattened_data))[0]
                valid_indices_list.append(set(valid_indices))

        # Make all raster valid index the same
        raster_valid_index = list(set.intersection(*valid_indices_list))

        # Filter the decision matrix to only include common valid values between raster
        # Transpose the decision matrix to row as alternatives
        decision_matrix = np.array(decision_matrix)[:, raster_valid_index].T

        objectives = [1 if self.criteria_direction[raster_file] 
                      else -1 for raster_file in self.raster.keys()]
        weights = [self.criteria_weights[raster_file] 
                   for raster_file in self.raster.keys()]

        return decision_matrix, weights, objectives, raster_valid_index

    def process_pymcdm(self):
        """
            Transform the data for pymcdm package.
        """

        decision_matrix = []
        valid_indices_list = [] 
        
        for raster_file, raster_path in self.raster.items():
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1).astype(float)

                # Debugging print
                print(f"Raster: {raster_file}, Min Value: {np.nanmin(raster_data)}, Max Value: {np.nanmax(raster_data)}")

                # Handle extreme negative values and invalid values
                extreme_values = [-1.7976931348623157e+308, -3.4028234663852886e+38]
                for extreme in extreme_values:
                    raster_data[raster_data == extreme] = np.nan
                raster_data[np.isinf(raster_data)] = np.nan

                # Get the index for valid cells
                flattened_data = raster_data.reshape(-1)
                decision_matrix.append(flattened_data)

                valid_indices = np.where(~np.isnan(flattened_data))[0]
                valid_indices_list.append(set(valid_indices))

        # Ensure all rasters have the same valid indices
        raster_valid_index = list(set.intersection(*valid_indices_list))
        
        # Filter the decision matrix to only include common valid values between raster
        # Transpose the decision matrix to row as alternatives
        decision_matrix = np.array(decision_matrix)[:, raster_valid_index].T

        # Objectives - True for maximization and False for minimization
        objectives = [self.criteria_direction[raster_file] for raster_file in self.raster.keys()]
        objectives = ['max' if obj else 'min' for obj in objectives]
        
        weights = [self.criteria_weights[raster_file] for raster_file in self.raster.keys()]
        criteria_names = list(self.raster.keys())
        alternatives_names = [f"Alternative_{i}" for i in range(1, decision_matrix.shape[0] + 1)]

        return decision_matrix, weights, criteria_names, alternatives_names, objectives, raster_valid_index
    
    def process_fuzzy(self):
        """
            Transform the data for pyDecision package (for fuzzy operations).
        """

        decision_matrix = []
        valid_indices_list = [] 
        
        for raster_file, raster_path in self.raster.items():
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1).astype(float)

                print(f"Raster: {raster_file}, Min Value: {np.nanmin(raster_data)}, Max Value: {np.nanmax(raster_data)}")

                # Handle extreme negative values and invalid values
                extreme_values = [-1.7976931348623157e+308, -3.4028234663852886e+38]
                for extreme in extreme_values:
                    raster_data[raster_data == extreme] = np.nan
                raster_data[np.isinf(raster_data)] = np.nan

                # Get the index for valid cells
                flattened_data = raster_data.reshape(-1)
                decision_matrix.append(flattened_data)

                valid_indices = np.where(~np.isnan(flattened_data))[0]
                valid_indices_list.append(set(valid_indices))

        # Transpose and ensure all raster have the same valid index for later mapping
        raster_valid_index = list(set.intersection(*valid_indices_list))
        decision_matrix = np.array(decision_matrix)[:, raster_valid_index].T

        # True for maximization and False for minimization for Objectives
        objectives = [self.criteria_direction[raster_file] for raster_file in self.raster.keys()]
        objectives = ['max' if obj else 'min' for obj in objectives]
        
        # fuzzy weights
        weights = [self.range_weights[raster_file] for raster_file in self.raster.keys()]
        criteria_names = list(self.raster.keys())
        alternatives_names = [f"Alternative_{i}" for i in range(1, decision_matrix.shape[0] + 1)]

        return decision_matrix, weights, criteria_names, alternatives_names, objectives, raster_valid_index

    def weighted_sum(self, num_class = 5):
        """
            Calculate the suitability scores and reclassified scores using Weighted Sum with skcriteria.
            Note that the skcriteria weighted sum method does not accept minimised criteria so it is recommended
            to reverse the score of minimised criteria first.
        """
        
        print("\nThe following is Weighted Sum --------------------------------------------------------------")
        
        # Retrieve the transformed data from processing method
        decision_matrix, weights, objectives, valid_indices = self.process_skcriteria()
        weighted_sum = simple.WeightedSumModel()
        rank = weighted_sum.evaluate(DecisionMatrix(decision_matrix, weights=weights, objectives=objectives))
        score = rank.e_.score
                
        # Store the shape of the original raster
        temp_raster = list(self.raster.values())[0]
        with rasterio.open(temp_raster) as src:
            weighted_sum_score = np.full(src.shape, np.nan)
            
            # Assign the score to the index of the valid cells
            weighted_sum_score.ravel()[valid_indices] = score
            
            with rasterio.open('weighted_sum_scores.tif', 'w', driver='GTiff',
                            height=weighted_sum_score.shape[0],
                            width=weighted_sum_score.shape[1], 
                            count=1, dtype=str(weighted_sum_score.dtype),
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(weighted_sum_score, 1)
        
            # Reclassify the scores into num_classes
            min_val = np.nanmin(weighted_sum_score)
            max_val = np.nanmax(weighted_sum_score)
            interval = (max_val - min_val) / num_class
            reclassified_data = np.zeros(weighted_sum_score.shape, dtype=np.uint8)
            for i in range(num_class):
                reclassified_data[(weighted_sum_score >= (min_val + i * interval)) & 
                                (weighted_sum_score < (min_val + (i+1) * interval))] = i + 1

            # Assign NaN values to class 0
            reclassified_data[np.isnan(weighted_sum_score)] = 0

            # Save the reclassified raster
            with rasterio.open('reclass_weighted_sum_scores.tif', 'w', driver='GTiff',
                            height=reclassified_data.shape[0],
                            width=reclassified_data.shape[1], 
                            count=1, dtype=str(reclassified_data.dtype),
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(reclassified_data, 1)
     
        return weighted_sum_score, reclassified_data

    def topsis(self, num_class = 5):
        """
            Calculate the suitability scores and reclassified scores using TOPSIS with skcriteria.
            Although TOPSIS allows minimised criteria, it is recommended reversed calculating them to avoid
            error in calculation.
        """

        print("\nThe following is TOPSIS --------------------------------------------------------------------")
        
        # Retrieve the transformed data from process method
        decision_matrix, weights, objectives, valid_indices = self.process_skcriteria()

        topsis = TOPSIS(metric='euclidean')
        decision = topsis.evaluate(DecisionMatrix(decision_matrix, objectives, weights))

        # Get the similarity scores (topsis score)
        topsis_score = decision.e_.similarity

        # Store the shape of the original raster
        temp_raster_path = list(self.raster.values())[0] 
        with rasterio.open(temp_raster_path) as src:
            topsis_raster_scores = np.full(src.shape, np.nan)
            
            # Assign the score to the index of the valid cells
            topsis_raster_scores.ravel()[valid_indices] = topsis_score

            with rasterio.open('topsis_scores.tif', 'w', driver='GTiff',
                            height=topsis_raster_scores.shape[0], width=topsis_raster_scores.shape[1],
                            count=1, dtype=topsis_raster_scores.dtype,
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(topsis_raster_scores, 1)
                
            # Reclassify the scores into num_class
            min_val = np.nanmin(topsis_raster_scores)
            max_val = np.nanmax(topsis_raster_scores)
            interval = (max_val - min_val) / num_class
            reclassified_data = np.zeros(topsis_raster_scores.shape, dtype=np.uint8)
            for i in range(num_class):
                reclassified_data[(topsis_raster_scores >= (min_val + i * interval)) & 
                                (topsis_raster_scores < (min_val + (i+1) * interval))] = i + 1

            # Assign NaN values to class 0
            reclassified_data[np.isnan(topsis_raster_scores)] = 0

            # Save the reclassified raster
            with rasterio.open('reclass_topsis_scores.tif', 'w', driver='GTiff',
                            height=reclassified_data.shape[0],
                            width=reclassified_data.shape[1], 
                            count=1, dtype=str(reclassified_data.dtype),
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(reclassified_data, 1)

        return topsis_raster_scores, reclassified_data
    
    def ftopsis(self, num_class = 5, delta = 0.1, fuzzy_decision_matrix=None):
        """
            Calculate the suitability scores and reclassified scores using Fuzzy TOPSIS with pyFDM TFN.
            Note that the method default set the delta of fuzzy values as 0.1, you may adjust according 
            the decision_matrix according your case if your layers value incoporate fuzzy set too.
        """
        
        print("\nThe following is Fuzzy TOPSIS --------------------------------------------------------------------")
        
        # Get the fuzzy variables for processing methods
        decision_matrix, weights, criteria_names, alternatives_names, objectives, valid_indices = self.process_fuzzy()

        # Convert cells in the decision_matrix to fuzzy numbers using the delta assumed
        if fuzzy_decision_matrix is None:
            fuzzy_decision_matrix = np.array([
                [(value-delta, value, value+delta) for value in row]
                for row in decision_matrix
            ])

        # Convert objectives to the format for pyFDM
        criterion_type = np.array([1 if obj == 'max' else -1 for obj in objectives])
        weights = np.array(weights)
        
        # Use the fuzzy_topsis method from pyfdm
        f_topsis = methods.fTOPSIS()
        scores = f_topsis(fuzzy_decision_matrix, weights, criterion_type)
            
        # Map the scores back to the original raster shape
        temp_raster_path = list(self.raster.values())[0]
        with rasterio.open(temp_raster_path) as src:
            ftopsis_scores = np.full(src.shape, np.nan)
                
            # Map the scores to the valid index
            ftopsis_scores.ravel()[valid_indices] = scores

            with rasterio.open("fuzzy_topsis_scores.tif", 'w', driver='GTiff',
                            height=ftopsis_scores.shape[0], width=ftopsis_scores.shape[1],
                            count=1, dtype=ftopsis_scores.dtype,
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(ftopsis_scores, 1)

            # Reclassify the scores into num_class
            min_val = np.nanmin(ftopsis_scores)
            max_val = np.nanmax(ftopsis_scores)
            interval = (max_val - min_val) / num_class
            reclassified_data = np.zeros(ftopsis_scores.shape, dtype=np.uint8)
            for i in range(num_class):
                reclassified_data[(ftopsis_scores >= (min_val + i * interval)) & 
                                (ftopsis_scores < (min_val + (i+1) * interval))] = i + 1

            # Assign NaN values to class 0
            reclassified_data[np.isnan(ftopsis_scores)] = 0

            # Save the reclassified raster
            with rasterio.open('reclass_fuzzy_topsis_scores.tif', 'w', driver='GTiff',
                            height=reclassified_data.shape[0],
                            width=reclassified_data.shape[1], 
                            count=1, dtype=str(reclassified_data.dtype),
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(reclassified_data, 1)
                
        return ftopsis_scores, reclassified_data
    
    def vikor(self, num_class = 5):
        """
            Calculate the suitability scores and reclassified scores using VIKOR with pymcdm.
        """
        
        print("\nThe following is VIKOR --------------------------------------------------------------------")

        # Get the variables from the processing method
        decision_matrix, weights, criteria_names, alternatives_names, objectives, valid_indices = self.process_pymcdm()
        criteria_types = np.array([1 if obj == 'max' else -1 for obj in objectives])

        # Change the variables to numpy array
        decision_matrix = np.array(decision_matrix)
        weights = np.array(weights)
        criteria_types = np.array(criteria_types)

        vikor_body = VIKOR()
        q_values = vikor_body(decision_matrix, weights, criteria_types)

        # Normalize and Reverse the suitability scores
        min_q = min(q_values)
        max_q = max(q_values)
        q_values = [(max_q - q) / (max_q - min_q) for q in q_values]
        
        # Map the scores back to the original raster shape
        temp_raster_path = list(self.raster.values())[0]
        with rasterio.open(temp_raster_path) as src:
            vikor_scores = np.full(src.shape, np.nan)
            
            # Only get values if the cell is in valid indices
            vikor_scores.ravel()[valid_indices] = q_values

            with rasterio.open("vikor_scores.tif", 'w', driver='GTiff',
                            height=vikor_scores.shape[0], width=vikor_scores.shape[1],
                            count=1, dtype=vikor_scores.dtype,
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(vikor_scores, 1)

            # Reclassify the scores into num_class
            min_val = np.nanmin(vikor_scores)
            max_val = np.nanmax(vikor_scores)
            interval = (max_val - min_val) / num_class
            reclassified_data = np.zeros(vikor_scores.shape, dtype=np.uint8)
            
            # Calculate values for the separated classes
            for i in range(num_class):
                reclassified_data[(vikor_scores >= (min_val + i * interval)) & 
                                (vikor_scores < (min_val + (i+1) * interval))] = i + 1

            # Assign NaN values to class 0
            reclassified_data[np.isnan(vikor_scores)] = 0

            # Save the reclassified raster
            with rasterio.open('reclass_vikor_scores.tif', 'w', driver='GTiff',
                            height=reclassified_data.shape[0],
                            width=reclassified_data.shape[1], 
                            count=1, dtype=str(reclassified_data.dtype),
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(reclassified_data, 1)
                
        return vikor_scores, reclassified_data

    def fvikor(self, num_class = 5, delta = 0.1, fuzzy_decision_matrix=None):
        """
            Calculate the suitability score using Fuzzy VIKOR with pyFDM TFN.
            Note that the method default set the delta of fuzzy values as 0.1, you may adjust according 
            the decision_matrix according your case if your layers value incoporate fuzzy set too.
        """
        
        print("\nThe following is Fuzzy VIKOR --------------------------------------------------------------------")
        
        # Get the fuzzy variables for processing methods
        decision_matrix, weights, criteria_names, alternatives_names, objectives, valid_indices = self.process_fuzzy()

        # Convert cells in the decision_matrix to fuzzy numbers using the delta assumed
        if fuzzy_decision_matrix is None:
            fuzzy_decision_matrix = np.array([
                [(value-delta, value, value+delta) for value in row]
                for row in decision_matrix
            ])

        # Convert objectives to the format for pyFDM
        criterion_type = np.array([1 if obj == 'max' else -1 for obj in objectives])
        weights = np.array(weights)
        
        # Use the fuzzy vikor method from pyfdm
        f_vikor = methods.fVIKOR()
        scores = f_vikor(fuzzy_decision_matrix, weights, criterion_type)
            
        # Map the scores back to the original raster shape
        temp_raster_path = list(self.raster.values())[0]
        with rasterio.open(temp_raster_path) as src:
            fvikor_scores = np.full(src.shape, np.nan)
                
            # Map the scores to the valid index
            fvikor_scores.ravel()[valid_indices] = scores

            with rasterio.open("fuzzy_vikor_scores.tif", 'w', driver='GTiff',
                            height=fvikor_scores.shape[0], width=fvikor_scores.shape[1],
                            count=1, dtype=fvikor_scores.dtype,
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(fvikor_scores, 1)
                    
            # Reclassify the scores into num_class
            min_val = np.nanmin(fvikor_scores)
            max_val = np.nanmax(fvikor_scores)
            interval = (max_val - min_val) / num_class
            reclassified_data = np.zeros(fvikor_scores.shape, dtype=np.uint8)
            
            # Calculate values for the separated classes
            for i in range(num_class):
                reclassified_data[(fvikor_scores >= (min_val + i * interval)) & 
                                (fvikor_scores < (min_val + (i+1) * interval))] = i + 1

            # Assign NaN values to class 0
            reclassified_data[np.isnan(fvikor_scores)] = 0

            # Save the reclassified raster
            with rasterio.open('reclass_fuzzy_vikor_scores.tif', 'w', driver='GTiff',
                            height=reclassified_data.shape[0],
                            width=reclassified_data.shape[1], 
                            count=1, dtype=str(reclassified_data.dtype),
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(reclassified_data, 1)
                
        return fvikor_scores, reclassified_data
    
    def fedas(self, num_class = 5, delta = 0.1, fuzzy_decision_matrix=None):
        """
            Calculate the suitability score using Fuzzy EDAS with pyFDM TFN.
            Note that the method default set the delta of fuzzy values as 0.1, you may adjust according 
            the decision_matrix according your case if your layers value incoporate fuzzy set too.
        """
        
        print("\nThe following is Fuzzy EDAS --------------------------------------------------------------------")
        
        # Get the fuzzy variables for processing methods
        decision_matrix, weights, criteria_names, alternatives_names, objectives, valid_indices = self.process_fuzzy()

        # Convert cells in the decision_matrix to fuzzy numbers using the delta assumed
        if fuzzy_decision_matrix is None:
            fuzzy_decision_matrix = np.array([
                [(value-delta, value, value+delta) for value in row]
                for row in decision_matrix
            ])

        # Convert objectives to the format for pyFDM
        criterion_type = np.array([1 if obj == 'max' else -1 for obj in objectives])
        weights = np.array(weights)
        
        # Use the fuzzy edas method from pyfdm
        fedas = methods.fEDAS()
        scores = fedas(fuzzy_decision_matrix, weights, criterion_type)
            
        # Map the scores back to the original raster shape
        temp_raster_path = list(self.raster.values())[0]
        with rasterio.open(temp_raster_path) as src:
            fedas_scores = np.full(src.shape, np.nan)
                
            # Map the scores to the valid index
            fedas_scores.ravel()[valid_indices] = scores

            with rasterio.open("fuzzy_edas_scores.tif", 'w', driver='GTiff',
                            height=fedas_scores.shape[0], width=fedas_scores.shape[1],
                            count=1, dtype=fedas_scores.dtype,
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(fedas_scores, 1)
                
            # Reclassify the scores into num_class
            min_val = np.nanmin(fedas_scores)
            max_val = np.nanmax(fedas_scores)
            interval = (max_val - min_val) / num_class
            reclassified_data = np.zeros(fedas_scores.shape, dtype=np.uint8)
            
            # Calculate values for the separated classes
            for i in range(num_class):
                reclassified_data[(fedas_scores >= (min_val + i * interval)) & 
                                (fedas_scores < (min_val + (i+1) * interval))] = i + 1

            # Assign NaN values to class 0
            reclassified_data[np.isnan(fedas_scores)] = 0

            # Save the reclassified raster
            with rasterio.open('reclass_fuzzy_vikor_scores.tif', 'w', driver='GTiff',
                            height=reclassified_data.shape[0],
                            width=reclassified_data.shape[1], 
                            count=1, dtype=str(reclassified_data.dtype),
                            crs=src.crs, transform=src.transform) as dst:
                dst.write(reclassified_data, 1)
                    
        return fedas_scores, reclassified_data


def correlation_matrix(reclass_dict):
    """
    Compute and plot the correlation coefficient matrix for reclassified suitability map.
    """
    
    # Normalize each raster data in the dictionary
    for key, data in reclass_dict.items():
        valid_data = data[~np.isnan(data) and (data != 0) & (data != 1)]
        reclass_raster_dict[key] = valid_data.flatten()
    
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(reclass_dict)
    
    # Compute the correlation matrix
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='cividis', vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.show()
    
    return correlation_matrix


# def reclass_common_plot(folder_path, base_layer=None, same_value=None):
#     """
#     Plot cells if the raster has the input value. Then clip and plot the base map.
#     """
    
#     if not os.path.exists(folder_path):
#         print("The provided folder path does not exist.")
#         return
    
#     raster_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]

#     # Process the base shapefile
#     base_geom = None
#     if base_layer:
#         base_gdf = gpd.read_file(base_layer)
#         base_geom = base_gdf.geometry.unary_union
    
#     # Initialize a matrix with ones
#     with rasterio.open(os.path.join(folder_path, raster_files[0])) as src:
#         common_cells = np.ones(src.shape, dtype=bool)
#         raster_crs = src.crs
    
#     for raster_file in raster_files:
#         with rasterio.open(os.path.join(folder_path, raster_file)) as src:
#             data = src.read(1)
#             # Update the common cells matrix
#             if same_value:
#                 value_mask = (data == same_value)
#                 common_cells &= value_mask

#     if not np.any(common_cells):
#         print("There are no common cells with the value", same_value, "across all rasters.")
#         return

#     # Reproject the shapefile to match the raster's CRS if they are different
#     if base_layer and base_gdf.crs != raster_crs:
#         base_gdf = base_gdf.to_crs(raster_crs)

#     # Plot the common cells
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.imshow(common_cells, cmap='Blues', interpolation='none')
    
#     # Plot the base layer shapefile
#     if base_layer:
#         base_gdf.boundary.plot(ax=ax, color='black', linewidth=1)
    
#     ax.axis('off')
#     plt.tight_layout()
#     plt.show()
    

if __name__ == "__main__":
    reclass_common_plot("reclass_suitability", 
                            base_layer="hk_wind_turbine_site_selection_case_study/study_area/study_area_without_constraints.shp", 
                            same_value=5)