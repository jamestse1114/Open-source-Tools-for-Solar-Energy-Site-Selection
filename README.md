# Python-based Open-source Tools for Renewable Energy Site Selection

## Description
This repository contains open-source Python tools for multi-criteria decision analysis (MCDA) of potential renewable energy sites. The tools use various geospatial data layers to assess the suitability of different locations for renewable energy installations, such as wind turbines. 

A case study of wind turbine site selection in Hong Kong is used to demonstrate the application of these tools. It utilizes criteria such as wind speed, elevation, roughness, slope and proximity to settlements.

The decision-making process involves the Analytic Hierarchy Process (AHP), Technique for Order Preference by Similarity to Ideal Solution (TOPSIS), ELimination Et Choix Traduisant la REalité (ELECTRE), and VlseKriterijumska Optimizacija I Kompromisno Resenje (VIKOR) methods. These methods use pairwise comparisons of alternatives to rank or select alternatives. 

## Installation
To run the tools, you will need to have Python installed on your computer. Additionally, you will need to install several Python packages. You can install the packages with the following command:

```python
pip install numpy pandas geopandas matplotlib rasterio shapely scipy
```

## Usage

To use these tools, you will need to provide geospatial data layers in the form of raster files. Each raster file should represent a criterion used in the decision-making process. The tools will prompt you for input regarding the weights of the different criteria.

After running the tools, the output will be a set of raster files representing the suitability scores of different locations for renewable energy installation based on each MCDA method.


To start using it, you'll need to install Python and some essential libraries. Please refer to the [Installation Guide](#) for detailed instructions. Once you have the prerequisites installed, you can download or clone this repository.
