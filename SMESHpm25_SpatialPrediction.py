"""
This python code generates daily PM2.5 spatial prediction for a given month and year, based on S-MESH ML framework
Inputs: XGBoost saved models
	All the spatial input parameters
	Training data for power transformation
	Year and month of interest

Outputs: Daily tif files with PM2.5 prediction over Europe
Further details can be obtained from related research article https://doi.org/10.1016/j.envres.2024.120363

"""

year='2021'
month='04'

xgbmodelpathFA = "/xnilu_wrk/users/sshe/AODproject/models/Final/XGBModelCAMSForAODnulls.model"
xgbmodelpathF = "/xnilu_wrk/users/sshe/AODproject/models/Final/XGBModelCAMSFornulls.model"
xgbmodelpathC = "/xnilu_wrk/users/sshe/AODproject/models/Final/XGBModelComP.model"
xgbmodelpathCC = "/xnilu_wrk/users/sshe/AODproject/models/Final/XGBModelCom.model"


#inputfile='/xnilu_wrk/users/sshe/AODproject/datasets/CAMS/DailyMeans/'+year+'/tifGrid/PMFtoGrid_'+year+month+'*.tif'
inputfile='/xnilu_wrk/users/sshe/AODproject/datasets/VIIRSDB_L2_AOD/AERDB_L2_VIIRS_NOAA20/SpatialData/DailyMeans/'+year+'/tifGrid/AOD/noaaAODtoGrid_'+year+month+'*.tif'
xgbpredictFile = '/xnilu_wrk/users/sshe/AODproject/datasets/Europe/Outputs/PM25/'

ext = 'tif'
# Basic libraries for reading files and handling datasets
import pandas as pd
import numpy as np
import glob
import os
import numpy.ma as ma
import datetime

# Library to read and save tif files
import rasterio

from rasterio.transform import Affine

import rioxarray as ri

import xarray as xr

# Machine learning related libraries
import xgboost as xgb
import joblib

# Multiprocessing libraries
#import multiprocessing
import time


# Normalisation functions
from sklearn.preprocessing import PowerTransformer


os.makedirs(os.path.dirname(xgbpredictFile),exist_ok=True)


### Generate tiff file from predictions stored in dataframe

def predictionToTif(mpredictions, refTif, fpath, scenario):
    "Transform back the log"
    mpredictions = np.expm1(mpredictions)

    predictions = pd.DataFrame({'pred': mpredictions})
    final = predictions
    print("Predictions Ready", predictions)
    outputf = xgbpredictFile

    "Prepare to save into file"

    mergedData = refTif
    xmin = float(mergedData.spatial_ref.attrs.get('GeoTransform').split(" ")[0])
    res = float(mergedData.spatial_ref.attrs.get('GeoTransform').split(" ")[1])
    ymax = float(mergedData.spatial_ref.attrs.get('GeoTransform').split(" ")[3])
    ## Define transform
    transform = Affine.translation(xmin, ymax) * Affine.scale(res, res * -1)

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

    predictedArray = np.array(final).reshape(meta.get('height'), meta.get('width'))

    lulc = ri.open_rasterio('/xnilu_wrk/users/sshe/AODproject/datasets/LULC/Corine1kmModennwoTK.tif')

    masked_output = ma.MaskedArray(predictedArray, mask=(np.isnan(lulc[0].values) | (lulc[0].values == 12) | (lulc[0].values == 0)))

    masked_output = masked_output.filled(np.nan)
    print("Masked output", masked_output)
    print("Masked output shape", masked_output.shape)

    if not os.path.exists(outputf):
        os.makedirs(outputf)

    outfilename = outputf + fpath+'/'+year+'/'+month+'/'+ 'EU_PM2P5'+ '_' + when+ '_' + scenario  + '.' + ext
    print("Masking done")

    with rasterio.open(outfilename, "w", **meta) as dst:
        band = masked_output
        dst.write(band, 1)
        dst.set_band_description(1, "prediction")
        dst.close()





"""   Open each file and extract all the values -> then store in dataframe """
def read_data(when):

    """ Reads all the input file data from corresponding path based on
    'when' parameter
    """

    fpath = "/xnilu_wrk/users/sshe/AODproject/datasets/InputParams/SpatialTest/NonTemporalInputs/ADEM1kmRegrid.tif"
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()

    df = pd.DataFrame({'DEM': sevD})


    fpath = "/xnilu_wrk/users/sshe/AODproject/datasets/InputParams/SpatialTest/NonTemporalInputs/ADEMStd.tif"
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()

    df['DEM_Std5km'] = sevD


    fpath = "/xnilu_wrk/users/sshe/AODproject/datasets/InputParams/SpatialTest/NonTemporalInputs/PopDen1kmGrid.tif"
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()

    df['popDen'] = sevD

    ndviWhen = datetime.datetime.strptime(when, '%Y%m%d').strftime('%Y-%m')
    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/VIIRS_NDVI/tifGrid/NDVIMean_' +ndviWhen + '-01' + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()
    df['NDVIMean'] = sevD


    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/ERA5/DailyMeans/' + year + '/PBH/tifGrid/eraPBHtoGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()
    df['blh'] = sevD


    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/ERA5/DailyMeans/' + year + '/ERA5Land/tifGrid/t2m/eraT2MtoGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()
    df['t2m'] = sevD


    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/ERA5/DailyMeans/' + year + '/ERA5Land/tifGrid/u10/eraU10toGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()
    df['u10'] = sevD

    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/ERA5/DailyMeans/' + year + '/ERA5Land/tifGrid/v10/eraV10toGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()
    df['v10'] = sevD


    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/ERA5/DailyMeans/' + year + '/ERA5Land/tifGrid/tp/eraTPtoGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()
    df['tp'] = sevD


    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/ERA5/DailyMeans/' + year + '/ERA5Land/tifGrid/sp/eraSPtoGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()
    df['sp'] = sevD


    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/ERA5/DailyMeans/' + year + '/ERA5Land/tifGrid/d2m/eraD2MtoGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()
    df['d2m'] = sevD

    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/VIIRSDB_L2_AOD/AERDB_L2_VIIRS_NOAA20/SpatialData/DailyMeans/' + year + '/tifGrid/AE/noaaAEtoGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()
    df['Angstrom_Exponent_Land_Best_Estimate'] = sevD

    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/VIIRSDB_L2_AOD/AERDB_L2_VIIRS_NOAA20/SpatialData/DailyMeans/' + year + '/tifGrid/AOD/noaaAODtoGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    noaa = sevD
    sevD = sevD.flatten()
    df['Aerosol_Optical_Thickness_550_Land_Best_Estimate'] = sevD

    df['DayOfYear'] = pd.Period(when).day_of_year

    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/CAMS/DailyMeans/' + year + '/tifGrid/PMFtoGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    sevD = sev.values
    sevD = sevD.flatten()
    df['pm2p5_conc'] = sevD

    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/CAMS/Analysis/DailyMeans/' + year + '/tifGrid/DailyM_PMAtoGrid_' + when + '.' + ext
    sev = ri.open_rasterio(fpath)
    # sev = sev.rio.clip_box(minx=1, miny=46, maxx=3.5, maxy=50)
    sevD = sev.values
    sevD = sevD.flatten()
    df['Ana_PM2P5'] = sevD

    aaiWhen = datetime.datetime.strptime(when, '%Y%m%d').strftime('%Y-%m-%d')
    fpath = '/xnilu_wrk/users/sshe/AODproject/datasets/S5P_AAI/DataFiles/tifGrid/AAItoGrid_' + aaiWhen + '.' + ext
    sev = ri.open_rasterio(fpath)
    # sev = sev.rio.clip_box(minx=1, miny=46, maxx=3.5, maxy=50)
    sevD = sev.values
    sevD = sevD.flatten()
    df['absorbing_aerosol_index'] = sevD


    print("Dataframe creation", df.head())
    print("Dataframe columns", df.columns)

    u = df['u10']
    v = df['v10']

    df['windSpeed'] = np.sqrt((u * u) + (v * v))
    df['windDirection'] = np.mod(np.rad2deg(np.arctan2(u, v)), 360)

    return df,sev




def preprocess(df):
    """ Apply transformations where applicable and build input data"""
    fcolumns = [

        'pm2p5_conc',
        #'Ana_PM2P5',
        'Aerosol_Optical_Thickness_550_Land_Best_Estimate',  'Angstrom_Exponent_Land_Best_Estimate',

        'NDVIMean', 'DEM', 'DEM_Std5km',    'absorbing_aerosol_index',

        'windSpeed', 'windDirection',  'sp', 'tp', 'd2m', 't2m' , 'blh',
        'DayOfYear',
        'popDen']

    xdf = df[fcolumns]
    print("Dataset to predict on", df.head())

    print("Data Transformation here = Logs and sqrts. After transformation")
    ### log transformed data

    logTrans = [
         'pm2p5_conc',
         #'Ana_PM2P5',
        'blh',
        'windSpeed',
        'tp',
        ]

    powTrans = ['sp', 'DEM', 'DEM_Std5km', 'absorbing_aerosol_index',
                'Aerosol_Optical_Thickness_550_Land_Best_Estimate', 'popDen']

    sqTrans = ['Angstrom_Exponent_Land_Best_Estimate']

    xdf.loc[:, logTrans] = np.log1p(xdf[logTrans])
    xdf.loc[:,sqTrans] = np.power(xdf[sqTrans], 2)
    xdf.loc[:, 'DayOfYear'] = (xdf['DayOfYear'] - 0.9) / (366 - 0.9)
    xdf.loc[:, 'NDVIMean'] = xdf['NDVIMean'] / 10000


    ## Generate power transform from training data
    xdf[powTrans] = np.float64(xdf[powTrans])
    # xdf['sp'] = yeojohnTr.fit_transform(xdf['sp'].values.reshape(-1,1))
    xdf[powTrans] = yeojohnTr.transform(xdf[powTrans])
    print("After Transformation",xdf)
    return xdf




def performML(xdf,refTif):

    """Perform Prediction using transform dataframe xdf which contains required data"""

    print("Prediction")
    pred_map = xgb.DMatrix(xdf.values)
    print("Inputs for prediction", pred_map)

    ### Take care of predictions here - total 3 models in working - One with Forecast, One with Forecast+AOD. These are base models. then the last one combines their inputs - meta model

    xgbFA = xgb.Booster()
    xgbFA.load_model(xgbmodelpathFA)
    FApredictions = xgbFA.predict(pred_map)
    print("First Prediction", FApredictions)

    # 2ND MODEL
    xxdf = xdf.drop(
        columns=['Aerosol_Optical_Thickness_550_Land_Best_Estimate', 'Angstrom_Exponent_Land_Best_Estimate'])
    predf_map = xgb.DMatrix(xxdf.values)
    print("Inputs for NON AOD prediction", predf_map)

    xgbF = xgb.Booster()
    xgbF.load_model(xgbmodelpathF)
    Fpredictions = xgbF.predict(predf_map)
    print("Second Prediction", Fpredictions)


    #3RD Model
    xgbC = xgb.Booster()
    xgbC.load_model(xgbmodelpathC)
    tf = pd.concat([xdf['Aerosol_Optical_Thickness_550_Land_Best_Estimate'], pd.DataFrame({'ForAODP': FApredictions}),
                    pd.DataFrame({'ForP': Fpredictions}), xdf['pm2p5_conc']], axis=1)
    print("Combined dataframe okay?", tf.head())
    cpred_map = xgb.DMatrix(tf.values)
    Cpredictions = xgbC.predict(cpred_map)
    print("Third Prediction", Cpredictions)


    predictionToTif(FApredictions, refTif,'Beta/Famodel', 'mFA')
    predictionToTif(Fpredictions, refTif, 'Beta/Fmodel','mF')
    predictionToTif(Cpredictions, refTif, 'Combined','mFAF')
    #predictionToTif(CCpredictions, refTif, 'Combined','mFAFs') ##Single







files = sorted(glob.glob(inputfile))
print(files)

"""May be below section can be modularised!!! For later if needed"""
### Need this transform object for later. But its only one time action, So perform it here and use it in function preprocess()
powTrans = ['sp', 'DEM', 'DEM_Std5km', 'absorbing_aerosol_index',
            'Aerosol_Optical_Thickness_550_Land_Best_Estimate', 'popDen']
## Generate power transform from training data
modelData = pd.read_csv('/xnilu_wrk/users/sshe/AODproject/datasets/InputParams/TrData/2021_2022nonTransformedData.csv',
                        sep=';')
yeojohnTr = PowerTransformer(standardize=True)
tempS = yeojohnTr.fit(modelData[powTrans])


for f in files:
    print("\nNow file ", f)
    when = f.split('.')[0]
    when=when[-8:]

    df,refTif = read_data(when)  ## return dataframe and the one tiff file data required for later metadata creation
    xdf = preprocess(df)
    performML(xdf,refTif)


