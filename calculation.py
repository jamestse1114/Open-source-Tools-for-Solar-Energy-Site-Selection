import rasterio
import matplotlib.pyplot as plt
import numpy as np

class SiteSuitability:
    def __init__(self, raster_path):
        self.dataset = rasterio.open(raster_path)
        self.suitability = None

    def calculate_suitability(self):
        # Read the raster values into a 2D array
        raster_values = self.dataset.read(1)

        # Placeholder function for calculating suitability
        # Replace this with your actual calculation
        self.suitability = raster_values / raster_values.max()

    def classify_suitability(self, bins):
        # Classify the suitability scores into bins
        self.suitability = np.digitize(self.suitability, bins)

    def plot_suitability(self):
        # Create a plot of the suitability
        plt.imshow(self.suitability, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Suitability')
        plt.show()


site_suitability = SiteSuitability('path_to_your_raster_data')

# Calculate suitability
site_suitability.calculate_suitability()

# Classify suitability into bins
bins = [0, 0.2, 0.4, 0.6, 0.8, 1]  # replace with your actual bin edges
site_suitability.classify_suitability(bins)

# Plot suitability
site_suitability.plot_suitability()
