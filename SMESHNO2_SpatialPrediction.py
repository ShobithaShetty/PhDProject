#!/xnilu_wrk/users/sshe/NO2project/codes/envir/phdno2/bin/python3
"""
This python code generates daily NO2 spatial prediction for a given month and year, based on S-MESH ML framework. There is option to
generate tif file based on certain NO2 filtering limit (below variable 'dataFilter') or not. 
Inputs: XGBoost saved model
	All the spatial input parameters
	Training data for power transformation
	Year and month of interest

Outputs: Daily tif files with PM2.5 prediction over Europe

Further details can be obtained from related research article https://doi.org/10.1016/j.rse.2024.114321
"""


xgbmodelpath = "/xnilu_wrk/users/sshe/NO2project/datasets/models/XGBModelRPRFiltered4500.model"
inputfile = '/xnilu_wrk/users/sshe/NO2project/datasets/Timeseries/Europe/Inputs/EU/2020/05/*'
xgbpredictFile = '/xnilu_wrk/users/sshe/NO2project/datasets/Timeseries/Europe/Outputs/EU/2020/Prediction/05'


year = '2021'
month= '04'

dataFilter = 'limit05' # if the new limit < 0.000008 is applied based on review suggestions, to remove certain TROPOMI NO2 data
inputfile = '/xnilu_wrk/users/sshe/NO2project/datasets/Timeseries/Europe/Inputs/EU/'+year+'/'+month+'/*.tif'

if dataFilter == 'limit05':
    xgbmodelpath = "/xnilu_wrk/users/sshe/NO2project/datasets/models/XGBModelRPRFiltered4500.model"
    xgbpredictFile = '/xnilu_wrk/users/sshe/NO2project/datasets/Timeseries/Europe/Outputs/EU/'+year+'/Reprocessed/Prediction/'+month
    no2_files = '/xnilu_wrk/users/sshe/NO2project/datasets/Timeseries/Europe/Inputs/EU/s5p-no2/'+year+'/s5p-no2_'
    sza_files = '/xnilu_wrk/users/sshe/NO2project/datasets/Timeseries/Europe/Inputs/EU/s5p-sza/'+year+'/s5p-sza_'
else:
    xgbpredictFile = '/xnilu_wrk/users/sshe/NO2project/datasets/Timeseries/Europe/Outputs/EU/'+year+'/Prediction/'+month
    xgbmodelpath = "/xnilu_wrk/users/sshe/NO2project/datasets/models/XGBModelVIIRSCorrSeasonalSampCV.model"


# month = 3
# year = 2020
# day = 26
normalised = True
saveAllBands = False

logTrans = ['tropospheric_NO2_column_number_density',
             'avg_rad', 'elevation', 'DEM_Std5km', 'Planetary_Boundary_Layer_Height',
              'windSpeed'
           ]

sqrtTrans = ['solar_zenith_angle','temperature_2m']




# Basic libraries for reading files and handling datasets
import pandas as pd
import numpy as np
import glob
import os
import numpy.ma as ma

# Library to read and save tif files
import rasterio
from rasterio.transform import Affine
import rioxarray as ri
import xarray as xr

# Machine learning related libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import joblib

# Visualisations
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Normalisation functions
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Multiprocessing libraries
import multiprocessing
import time


# Load the models
xgbM = xgb.Booster()
xgbM.load_model(xgbmodelpath)
# rf = joblib.load(rfmodelpath)


#features = modelData.drop(columns=['datetimeS5P', 'WeighAvg', 'cloud_fraction']).columns
features = ['tropospheric_NO2_column_number_density',
 'solar_zenith_angle',
 'avg_rad',
 'elevation',
 'DEM_Std5km',
 'NDVIMean',
 'Planetary_Boundary_Layer_Height',
 'temperature_2m',
 'windSpeed',
 'windDirection',
 'DayOfYear',
 'weekDay']


# print(features)


### Define a function that reads each input file to be regressed and calls the ML model on it

def performML(inpF,suffix):
    """

    :param inpF: Single day spatial data with all parameters as bands
    :param suffix: whether its RPR or OFFL
    :Feeds data to another function that performs actual prediction.
    :Prepares data for ML prediction in the form dataframe and information required to save into a tif file
    """

    print("Inside Data Preparation")
    # Read the input tiff file
    mergedData = ri.open_rasterio(inpF)  # Provide the input file
    no2Data=xr.open_dataset(no2_files + suffix)
    szaData=xr.open_dataset(sza_files + suffix)

    ### Sort same as mergeData
    no2Data=no2Data.sortby('y',ascending=False)
    szaData=szaData.sortby('y',ascending=False)

    # Get the input data into dataframe for giving into the model
    print("Raster reading done")
    start_time = time.time()
    data = mergedData.to_dataframe(name='BandValues').unstack(level=0)['BandValues'].iloc[:].values
    print("Unstacking done")
    #print("Unstached dataframe size",data.memory_usage(deep=True))
    pred = pd.DataFrame(data, columns=['tropospheric_NO2_column_number_density', 'cloud_fraction', \
									   'solar_zenith_angle', 'avg_rad', \
                                       'elevation', 'DEM_Std5km', 'landcover', 'NDVIMean',
                                       'Planetary_Boundary_Layer_Height', \
                                       'temperature_2m',
                                       'u_component_of_wind_10m', 'v_component_of_wind_10m' \
                                       ])
    pred['tropospheric_NO2_column_number_density']=no2Data.band_data.values[0].flatten()
    pred['solar_zenith_angle'] = szaData.band_data.values[0].flatten()
    u = pred['u_component_of_wind_10m']
    v = pred['v_component_of_wind_10m']
    pred['windSpeed'] = np.sqrt(u * u + v * v )
    pred['windDirection'] = np.mod(np.rad2deg(np.arctan2(u ,v )),360)
    print("Dataframe creation done")
    print("New image dataframe",pred.memory_usage(deep=True))

    xmin = float(mergedData.spatial_ref.attrs.get('GeoTransform').split(" ")[0])
    res = float(mergedData.spatial_ref.attrs.get('GeoTransform').split(" ")[1])
    ymax = float(mergedData.spatial_ref.attrs.get('GeoTransform').split(" ")[3])
    ## Define transform
    transform = Affine.translation(xmin, ymax) * Affine.scale(res, res * -1)
    print("Transformation done")

    # Fetch the metadata and update with new and existing image information
    meta = mergedData.spatial_ref.attrs
    meta.update(count=1)  ## only save predicted variable
    meta.update(driver="GTiff")
    meta.update(width=mergedData.rio.width)
    meta.update(height=mergedData.rio.height)
    meta.update(dtype='float64')
    meta.update(crs=mergedData.rio.crs)
    meta.update(transform=transform)
    meta.update(nodata=np.nan)

    # Fetch date of the file - file pattern : EU_2019-02-01
    dates = inpF.split('.')[0]
    day = int(dates[-2:])
    month = int(dates[-5:-3])
    year = int(dates[-10:-6])

    df = pd.DataFrame({'year': [year], 'month': [month], 'day': [day]})
    study_date = pd.to_datetime(df)
    print("Inside ML for ", dates)

    if 'weekDay' in features:
        pred['weekDay'] = study_date.apply(lambda x: x.dayofweek > 4)[0]  # Only when including
    if 'DayOfYear' in features:
        pred['DayOfYear'] = study_date.dt.day_of_year[0]

    param_test = pred[features]
    if normalised == True:
       #param_test.loc[:, num_feat] = scaler.transform(param_test[num_feat])
       param_test.loc[:,logTrans] = np.log1p(param_test[logTrans])
       param_test.loc[:,sqrtTrans] = np.sqrt(param_test[sqrtTrans])

    print("Preprocessing done.calling ML method")
    end_time = time.time()
    print("Time for data preparation",end_time - start_time)

    s5p = no2Data.band_data.values[0]
    model_predict('XGB', param_test, meta, mergedData, s5p, dates[-13:])


def model_predict(modeltype, predictionData, metaD, originalImg, s5p, fileN):
    """
    Based on the given model, predict on the input file and save the results in a tiff file

    :param modeltype: XGB or RF
    :param predictionData: Dataframe with all inputs in the order from top-left to bottom-right
    :param metaD: Tif file metadata
    :param originalImg: Single day tif data for lulc information (for masking)
    :param s5p: S5P data for masking
    :param fileN: data for filename
    :return: No return but the saves a single day predicted file as tif
    """

    print("Inside Model Prediction")
    start_time = time.time()

    if modeltype == 'RF':
        predictionData = predictionData.fillna(0)
        pred_map = np.array(predictionData)
        mpredictions = rf.predict(pred_map)  # # xgbM.predict(pred_map)
        outputf = rfpredictFile
    elif modeltype == 'XGB':
        # print(predictionData.head())
        # pred_map = np.array(predictionData)
        pred_map = xgb.DMatrix(predictionData.values)
        # print("DMatrix")
        mpredictions = xgbM.predict(pred_map)  # # xgbM.predict(pred_map)
        outputf = xgbpredictFile

    print("Prediction is made ....")

    predictions = pd.DataFrame({'pred': mpredictions})
    final = predictions
    print("Predictions stored - done", final.head())
    if normalised == True:
        final.loc[:,'pred'] = np.expm1(final["pred"] )

    predictedArray = np.array(final).reshape(metaD.get('height'), metaD.get('width'))

    # Fetch masking data from S5P and LULC
    #s5p = originalImg[0]
    lulc = originalImg[6]

    if dataFilter == 'limit05':
        masked_output = ma.MaskedArray(predictedArray, mask=(
                np.isnan(lulc.values) | np.isnan(s5p) | (s5p == 0) | (s5p < 0.000008) | (lulc.values == 12)))
    else:
        masked_output = ma.MaskedArray(predictedArray, mask=(
                np.isnan(lulc.values) | np.isnan(s5p) | (s5p == 0) | (lulc.values == 12)))

    masked_output = masked_output.filled(np.nan)
    ext = 'tif'
    outfilename = f'{outputf}/{fileN}.{ext}'
    print("Masking done")

    with rasterio.open(outfilename, "w", **metaD) as dst:
        band = masked_output
        dst.write(band, 1)
        dst.set_band_description(1, "prediction")
        dst.close()


    print("Done ", fileN)
    end_time = time.time()

    print("Prediction and final tiff file storage",end_time - start_time)


if __name__ == "__main__":
    files = sorted(glob.glob(inputfile))
    print(files)

    if not os.path.exists(xgbpredictFile):
        os.makedirs(xgbpredictFile)

    # result = pool.map(performML,files)
    for f in files:
        suffix = f.split('/')[-1][3:]
        print("\nNow file ",f)
        performML(f,suffix)
