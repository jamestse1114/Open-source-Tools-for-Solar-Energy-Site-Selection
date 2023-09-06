# Python-based Open-source Tools for Renewable Energy Site Selection

## Description
This repository contains a unified workflow application tailored for GIS-MCDA integrated site selection, focusing primarily on renewable energy applications. The toolkit streamlines the GIS-MCDA process, enhancing efficiency, consistency, and replicability. It's designed to be user-friendly, making the complex process of site selection accessible to both experts and novices in the field.

## Installation
To run the tools, you will need to have Python installed on your computer. Additionally, you will need to install several Python packages. You can install the packages with the following command:

```python
pip install numpy pandas geopandas matplotlib rasterio shapely scipy
```

## Usage

To use these tools, you will need to provide geospatial data layers in the form of raster files. Each raster file should represent a criterion used in the decision-making process. The tools will prompt you for input regarding the weights of the different criteria.

After running the tools, the output will be a set of raster files representing the suitability scores of different locations for renewable energy installation based on each MCDA method.


To start using it, you'll need to install Python and some essential libraries. Please refer to the [Installation Guide](#) for detailed instructions. Once you have the prerequisites installed, you can download or clone this repository.
