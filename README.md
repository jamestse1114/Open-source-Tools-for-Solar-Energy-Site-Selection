# Python-based Open-source Tools for Renewable Energy Site Selection

## Description
This repository contains a unified workflow application tailored for GIS-MCDA integrated site selection, focusing primarily on renewable energy applications. The toolkit streamlines the GIS-MCDA process, enhancing efficiency, consistency, and replicability. It's designed to be user-friendly, making the complex site selection process accessible to experts and novices. If the user only requires using a specific part of the application, for example, the data transformation part, they can do so.

## Features

- **Geospatial Data Handling**: Comprehensive tools for transforming and processing geospatial data.
- **Unified Workflow**: A holistic platform that merges the capabilities of GIS and MCDA, eliminating the need for multiple applications.
- **Objective Decision Framework**: Minimising the reliance on subjective expert inputs by criteria weighting tools.
- **Multiple Weighting and MCDA methods**: Various MCDA methods are provided for the users for their specific needs.

## Installation
You must have at least Python 3.8 installed on your computer to run the tools. Additionally, you will need to install several Python packages. You can install the packages with the following command:

```python
pip install numpy pandas geopandas matplotlib rasterio shapely scipy skcriteria pymcdm pydecision pyds
```

## Usage

To use these tools, you must provide geospatial data layers as raster files. Each raster file should represent a criterion used in the decision-making process. The tools will prompt you for input regarding the weights of the different criteria.

After running the tools, the output will be a set of raster files representing the suitability scores of different locations for renewable energy installation based on each MCDA method.

You'll need to install Python and some essential libraries to start using it. Once the prerequisites are installed, you can download or clone this repository.
