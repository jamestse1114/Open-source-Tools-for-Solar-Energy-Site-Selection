import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import seaborn as sns
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


def get_directory():
    """
        Ask the user for the directory name and check if it exists.
        Return the project directory
    """

    directory = input("Enter the directory name: ")
    print(directory)

    if os.path.exists(directory) == False:
        print(f"Directory {directory} not found.")
        return None

    return directory


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


def normalise_raster(directory, raster):
    """
        Normalise a .TIF raster between 0 and 1 and save the result.
    """

    normalised_layer_folder = os.path.join(directory, "normalised_layers")
    if os.path.exists(normalised_layer_folder) != True:
        os.makedirs(normalised_layer_folder)

    if os.path.exists(raster) != True:
        print(f"{raster} does not exist.")
        return

    # Load the raster layer and information
    with rasterio.open(raster) as rast:
        data = rast.read(1)
        meta = rast.meta

    # Print the min and max values of the original raster data
    print(f"Original Min value: {np.min(data)}")
    print(f"Original Max value: {np.max(data)}")

    # Normalize the data
    data_min = np.min(data)
    data_max = np.max(data)
    normalized_data = (data - data_min) / (data_max - data_min)

    # Print the min and max values of the normalized raster data
    print(f"Normalized Min value: {np.min(normalized_data)}")
    print(f"Normalized Max value: {np.max(normalized_data)}")

    output_path = os.path.join(normalised_layer_folder, f'normalized_{os.path.basename(raster)}')
    with rasterio.open(output_path, 'w', **meta) as rest:
        rest.write(normalized_data, 1)

    print(f"Normalised raster saved at: {output_path}")
    
    
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
    plt.show()


def plot_raster(raster):
    """
        Display a raster file.
    """

    with rasterio.open(raster) as rast:
        data = rast.read(1)

    plt.imshow(data, cmap='viridis')
    plt.colorbar(label='Raster Value')
    plt.show()


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


def compute_correlation_matrix(folder_path):
    """
    Compute and plot the correlation coefficient matrix for .tif raster in the folder.
    """
    
    raster_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    raster_data_dict = {}
    
    # Load each raster and store its data in a dictionary
    for raster_file in raster_files:
        with rasterio.open(os.path.join(folder_path, raster_file)) as src:
            data = src.read(1)
            valid_data = data[~np.isnan(data)].flatten()  # Extract non-NaN values and flatten the array
            raster_data_dict[raster_file] = valid_data
    
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame(raster_data_dict)
    
    # Compute the correlation matrix
    corr_matrix = df.corr()
    
    # Plot the correlation matrix using Seaborn's heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix of Rasters")
    plt.show()
    
    return corr_matrix


def reclass_suitability(num_of_class, raster, study_area_shp=None):
    """
    Reclassify a raster based on the number of classes and if a study_area is input, the raster will also be clipped.
    Created to reclassify suitability layers.
    """
    
    with rasterio.open(raster) as src:
        raster_data = src.read(1)
        raster_meta = src.meta

    if study_area_shp:
        # Load the shapefile and transform its crs to that of the raster
        study_area = gpd.read_file(study_area_shp)
        study_area = study_area.to_crs(raster_meta['crs'])
        
        # Clip the raster using the shapefile
        outimg, outtrans = mask(src, study_area.geometry, crop=True)
        raster_data = outimg[0]
        raster_meta.update({"height": raster_data.shape[0],
                            "width": raster_data.shape[1],
                            "transform": outtrans})
    

    valid_data = raster_data[~np.isnan(raster_data)]
    normalised = (valid_data - valid_data.min()) / (valid_data.max() - valid_data.min())
    
    # reclassify to the number of classes
    reclassified_data = (normalised * num_of_class).astype(int)
    reclassified_data[reclassified_data == num_of_class] = num_of_class - 1
    
    # Map the reclassified values to the original raster shape but ignoring nan values
    raster_data[~np.isnan(raster_data)] = reclassified_data
    
    # Save the reclassified suitability score
    output_path = os.path.join(os.path.dirname(raster), "suitability_reclass")
    with rasterio.open(output_path, 'w', **raster_meta) as dst:
        dst.write(raster_data, 1)
    
    return raster_data


def plot_rasters_with_same_value(folder_path, base_layer=None, same_value=None):
    """
    Plot rasters from a folder. If all rasters have the same value for a cell, highlight that cell.
    Optionally, clip the rasters to a base layer shapefile.
    
    Parameters:
    - folder_path: Path to the folder containing the rasters.
    - base_layer: Path to the base layer shapefile for clipping (optional).
    - same_value: Value to check for across all rasters.
    """
    
    raster_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    
    # Load the base layer shapefile if provided
    if base_layer:
        base_gdf = gpd.read_file(base_layer)
        base_geometry = base_gdf.geometry
    
    fig, ax = plt.subplots(len(raster_files), 1, figsize=(10, 5 * len(raster_files)))
    
    for i, raster_file in enumerate(raster_files):
        with rasterio.open(os.path.join(folder_path, raster_file)) as src:
            
            # Clip the raster to the base layer if provided
            if base_layer:
                out_image, out_transform = mask(src, base_geometry, crop=True)
                with MemoryFile() as memfile:
                    with memfile.open(driver="GTiff", height=out_image.shape[1], 
                                      width=out_image.shape[2], count=1, dtype=out_image.dtype, 
                                      crs=src.crs, transform=out_transform) as mem_raster:
                        mem_raster.write(out_image[0], 1)
                        data = mem_raster.read(1)
            else:
                data = src.read(1)
            
            # If same_value is provided, create a mask where all rasters have the same value
            if same_value:
                mask = np.ones_like(data, dtype=bool)
                for rf in raster_files:
                    with rasterio.open(os.path.join(folder_path, rf)) as r:
                        r_data = r.read(1)
                        mask &= (r_data == same_value)
                data[~mask] = np.nan  # Set all other values to NaN so they won't be plotted
            
            ax[i].imshow(data, cmap='viridis')
            ax[i].set_title(raster_file)
            ax[i].axis('off')
    
    plt.tight_layout()
    plt.show()


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

        # Save the criteria importance and criteria direction from user's input
        for raster_file in self.raster.keys():
            fuzzy_values = input(f"Importance values for '{raster_file}' (e.g., 4,5.3,6): ").split(',')
            criteria_importance.append((float(fuzzy_values[0]), float(fuzzy_values[1]), float(fuzzy_values[2])))
            direction = input(f"Is '{raster_file}' a maximization criterion? (yes/no): ").strip().lower()
            criteria_direction[raster_file] = direction == 'yes'

        # Compute a fuzzy pairwise comparison matrix
        fuzzy_matrix = []
        for i in range(criteria_num):
            row = []
            for j in range(criteria_num):
                if i == j:
                    row.append((1, 1, 1))
                else:
                    row.append(criteria_importance[j])
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
        Bayesian theory to process and output a confidence interval of the weight of criteria.
        
        Parameters:
        - criteria_weights_list: List of dictionaries containing experts' weights for each criterion.
        - uncertainty_factor: Optional float representing the uncertainty in each expert's opinion. 
                            If not provided, a default value of 0.1 (10%) is used.
        """
        
        # If no uncertainty factor is provided, use a default value of 0.1
        if uncertainty_factor is None:
            uncertainty_factor = 0.1

        # Initialize an empty list to store mass functions
        mass_functions = []

        # Process each expert's opinion
        for expert_weights in criteria_weights_list:
            # Create a new mass function for the expert
            mf = MassFunction()
            
            # Calculate the total mass based on the expert's weights
            total_mass = 0
            for criterion, weight in expert_weights.items():
                # Adjust the weight based on the uncertainty factor
                adjusted_weight = weight * (1 - uncertainty_factor)
                mf[frozenset([criterion])] = adjusted_weight
                total_mass += adjusted_weight
            
            # Assign the remaining mass to represent the expert's uncertainty
            mf[frozenset(expert_weights.keys())] = 1 - total_mass
            
            # Add the mass function to the list
            mass_functions.append(mf)

        # Combine the mass functions from all experts
        combined_mf = mass_functions[0]
        for next_mf in mass_functions[1:]:
            combined_mf = combined_mf & next_mf

        # Calculate belief and plausibility values
        beliefs = combined_mf.bel()
        plausibilities = combined_mf.pl()

        # Extract the range weights and criteria weights
        range_weights = {}
        criteria_weights = {}
        for criteria_set, belief_value in beliefs.items():
            if len(criteria_set) == 1:  # We're only interested in individual criteria
                criterion_name = list(criteria_set)[0]
                lower_bound = belief_value
                upper_bound = plausibilities[criteria_set]
                average_weight = (lower_bound + upper_bound) / 2

                range_weights[criterion_name] = (lower_bound, average_weight, upper_bound)
                criteria_weights[criterion_name] = average_weight

        print(range_weights)
        print(criteria_weights)
        
        return range_weights, criteria_weights

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

    def weighted_sum(self):
        """
            Calculate the suitability score using Weighted Sum with skcriteria.
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

        return weighted_sum_score

    def topsis(self):
        """
            Calculate the suitability score using TOPSIS with skcriteria.
            Although TOPSIS allows minimised criteria, it is recommended to reverse calculating them to avoid
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

        return topsis_raster_scores
    
    def ftopsis(self, delta = 0.1, fuzzy_decision_matrix=None):
        """
            Calculate the suitability score using Fuzzy TOPSIS with pyFDM TFN.
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
                    
        return ftopsis_scores
    
    def vikor(self):
        """
            Calculate the suitability score using VIKOR with pymcdm.
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

        return vikor_scores

    def fvikor(self, delta = 0.1, fuzzy_decision_matrix=None):
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
                    
        return fvikor_scores
    
    def fedas(self, delta = 0.1, fuzzy_decision_matrix=None):
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
                    
        return fedas_scores
    
    
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

    expert_op = [
    {'clipped_elevation.tif': 0.16903374801527896, 'clipped_river.tif': 0.16132755383444564, 'clipped_roughness.tif': 0.1732240982367213, 'clipped_settlements.tif': 0.16497082699188934, 'clipped_slope.tif': 0.16497082699188934, 'clipped_wind_speed.tif': 0.1664729459297755},
    {'clipped_elevation.tif': 0.15903374801527896, 'clipped_river.tif': 0.17132755383444564, 'clipped_roughness.tif': 0.1632240982367213, 'clipped_settlements.tif': 0.17497082699188934, 'clipped_slope.tif': 0.15497082699188934, 'clipped_wind_speed.tif': 0.1764729459297755},
    {'clipped_elevation.tif': 0.17903374801527896, 'clipped_river.tif': 0.15132755383444564, 'clipped_roughness.tif': 0.1832240982367213, 'clipped_settlements.tif': 0.15497082699188934, 'clipped_slope.tif': 0.17497082699188934, 'clipped_wind_speed.tif': 0.1564729459297755},
    {'clipped_elevation.tif': 0.14903374801527896, 'clipped_river.tif': 0.18132755383444564, 'clipped_roughness.tif': 0.1532240982367213, 'clipped_settlements.tif': 0.18497082699188934, 'clipped_slope.tif': 0.14497082699188934, 'clipped_wind_speed.tif': 0.1864729459297755},
    {'clipped_elevation.tif': 0.12903225806451618, 'clipped_river.tif': 0.1612903225806451, 'clipped_roughness.tif': 0.03225806451612904, 'clipped_settlements.tif': 0.19354838709677416, 'clipped_slope.tif': 0.22580645161290325, 'clipped_wind_speed.tif': 0.2580645161290323},
    {'clipped_elevation.tif': 0.1, 'clipped_river.tif': 0.3, 'clipped_roughness.tif': 0.3, 'clipped_settlements.tif': 0.1, 'clipped_slope.tif': 0.1, 'clipped_wind_speed.tif': 0.1}
    ]

    gis_mcda.dst(expert_op)

