''' 
All libraries imported
'''
import glob, sys, os
import netCDF4 as nc
import pandas as pd
import pickle
import datetime
from mpl_toolkits.basemap import Basemap
from scipy.stats import norm
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from scipy.stats import pearsonr
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

'''
Global Variables
'''
LEARNING_RATE_BASE = 0.0001
BATCH_SIZE = 16
NEPO = 500
VAL_RATIO = 0.1
LEARNING_RATE_DECAY = 0.99
DECAY_STEPS = 30
NREC = 266
HEIGHT = 96
WIDTH = 96
NBANDS = 22

'''
Predetermined filepaths under Bamboo linux subsystem.
Note: These filepaths can be changed as suitable.
'''
image_path = 'C:/Users/fzlce/Documents/VSCodeProjects/UNet2023/OldModel/'
modelfile =image_path +'all_models_final/mm.best.hdf5'
logfile = image_path +'all_models_final/log.csv'
hist_file =image_path +'all_models_final/myModelUNet5_history.pl'
eval_file =image_path +"all_models_final/eval_data.dat.npz"

'''
Helper functions
'''
def l2norm(arr1, arr2):
    result = np.sqrt(np.square(arr1) + np.square(arr2))
    return result

def s_filter(arr):
    arr[arr > 1] = 2
    return arr

def normalize(arr, var):
    scaled = var.fit_transform(arr)
    return scaled

def t_normalize(arr, var):
    scaled = var.transform(arr)
    return scaled

'''
Preprocessing step
'''
def preprocess(fp):
    dir = glob.glob(fp)
    BT = np.empty([0,HEIGHT,WIDTH, NBANDS])
    lat = np.empty([0,HEIGHT,WIDTH,1])
    lon = np.empty([0,HEIGHT,WIDTH,1])
    ps = np.empty([0,HEIGHT,WIDTH,1])
    stype = np.empty([0,HEIGHT, WIDTH, 1])
    clw = np.empty([0,HEIGHT, WIDTH, 1])
    u_ws = np.empty([0,HEIGHT,WIDTH,1])
    v_ws = np.empty([0,HEIGHT,WIDTH,1])
    ZA = np.empty([0,HEIGHT,WIDTH,1])

    indicies = []
    count = 1
    print(datetime.datetime.now())
    for f in dir:
        ds = nc.Dataset(f)
        if np.all(ds.variables['stype'][:] ==1):
            indicies.append(count)
            BT = np.append(BT, np.reshape(ds.variables['BT'][:], (1, HEIGHT, WIDTH, NBANDS)), axis = 0)
            lat = np.append(lat, np.reshape(ds.variables['Latitude'][:],(1, HEIGHT, WIDTH,1)), axis = 0)
            lon = np.append(lon, np.reshape(ds.variables['Longitude'][:],(1, HEIGHT, WIDTH,1)), axis = 0)
            u_ws = np.append(u_ws, np.reshape(ds.variables['wind_u10'][:],(1, HEIGHT, WIDTH,1)), axis = 0)
            v_ws = np.append(v_ws, np.reshape(ds.variables['wind_v10'][:],(1, HEIGHT, WIDTH,1)), axis = 0)
            clw =np.append(clw, np.reshape(ds.variables['clw'][:],(1, HEIGHT, WIDTH,1)), axis = 0)
            ZA = np.append(ZA, np.reshape(ds.variables['Zenith_Angle'][:],(1, HEIGHT, WIDTH,1)), axis = 0)
            ps = np.append(ps, np.reshape(ds.variables['surface_pressure'][:],(1, HEIGHT, WIDTH,1)), axis = 0)
        count+=1
    
    ws = l2norm(u_ws, v_ws)
    ps/=100
    #stype = s_filter(stype)
    inputs = np.concatenate((lat, lon, ZA, BT), axis = -1)
    labels = np.concatenate((ws, ps), axis = -1)

    #Sample indicies (WE want the test indicies for the min PS max WS calculation
    indicies = np.array(indicies)
    inputs, labels, clw, indicies = shuffle(inputs, labels, clw,indicies, random_state=10)
    # Train Test Split: Set test set ratio to be 0.1
    train_x, test_x, train_y, test_y, train_idx, test_idx = train_test_split(inputs, labels, indicies, test_size=0.1, random_state=10)
    # Do the same for the clw array
    train_clw, test_clw = train_test_split(clw, test_size=0.1, random_state= 10)
    #Keep clw indicies > 0.3
    indiciesclw = list(zip(*np.where(test_clw <= 0.3)))
    indiciesclw = [t[:-1] for t in indiciesclw]
    #Convert clw_indices to a list of arrays for advanced indexing
    indiciesclw = [np.array(indices) for indices in zip(*indiciesclw)]

    shape_x_i = train_x[:,:,:,0].shape
    shape_x_ti = test_x[:,:,:,0].shape
    shape_y_ti = test_y[:,:,:,0].shape
    shape_y_i = train_y[:,:,:,0].shape

    #normalize x_train and y_train
    train_x_norm = np.empty(train_x.shape[0:3]+(0,))
    test_x_norm = np.empty(test_x.shape[0:3]+(0,))
    for i in range(0,25):
        train2Dxi = train_x[:,:,:,i].reshape(-1,1)
        test2Dxi = test_x[:,:,:,i].reshape(-1,1)
        scaler_xi = StandardScaler()
        X_scaled_i = normalize(train2Dxi, scaler_xi)
        X_scaled_ti = t_normalize(test2Dxi, scaler_xi)
        train_x_norm_i = np.expand_dims(X_scaled_i.reshape(shape_x_i), axis = -1)
        train_x_norm = np.concatenate((train_x_norm, train_x_norm_i), axis = -1)
        test_x_norm_i = np.expand_dims(X_scaled_ti.reshape(shape_x_ti), axis = -1)
        test_x_norm = np.concatenate((test_x_norm, test_x_norm_i), axis = -1)

     #normalize y_train
    train2Dy_ws = train_y[:,:,:,0].reshape(-1,1)
    train2Dy_ps = train_y[:,:,:,1].reshape(-1,1)
    scaler_y_ws = StandardScaler()
    Y_scaled_ws = normalize(train2Dy_ws, scaler_y_ws)
    scaler_y_ps = RobustScaler()
    Y_scaled_ps = normalize(train2Dy_ps, scaler_y_ps)

    # Reshape the scaled data back to 4D
    train_y_norm_ws = np.expand_dims(Y_scaled_ws.reshape(shape_y_i), axis = -1)
    train_y_norm_ps = np.expand_dims(Y_scaled_ps.reshape(shape_y_i), axis = -1)
    train_y_norm = np.concatenate((train_y_norm_ws, train_y_norm_ps), axis = -1)

    #normalize y_test
    test2Dy_ws = test_y[:,:,:,0].reshape(-1,1)
    test2Dy_ps = test_y[:,:,:,1].reshape(-1,1)
    Y_tscaled_ws = t_normalize(test2Dy_ws, scaler_y_ws)
    Y_tscaled_ps = t_normalize(test2Dy_ps, scaler_y_ps)

    # Reshape the scaled data back to 4D
    test_y_norm_ws = np.expand_dims(Y_tscaled_ws.reshape(shape_y_ti), axis = -1)
    test_y_norm_ps = np.expand_dims(Y_tscaled_ps.reshape(shape_y_ti), axis = -1)
    test_y_norm = np.concatenate((test_y_norm_ws, test_y_norm_ps), axis = -1)
    print(datetime.datetime.now())

    return train_x_norm, train_y_norm, test_x_norm, test_y_norm, train_x, train_y, test_x, test_y, scaler_y_ws, scaler_y_ps, indiciesclw, test_idx, test_clw

'''
Here is the custom UNet architecture
Takes in inputs of 96 * 96 * 25
Pretty neat!
'''
def model_architecture(shape):
    inputs = Input(shape)

    #Build the encoder (contraction path)
    c1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(inputs)
    c1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(p1)
    c2 = Conv2D(128, 3, activation = 'relu', padding = 'same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(p2)
    c3 = Conv2D(256, 3, activation = 'relu', padding = 'same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)

    c4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(p3)
    c4 = Conv2D(512, 3, activation = 'relu', padding = 'same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    base = Conv2D(1024, 3, activation = 'relu', padding = 'same')(p4)
    base = Conv2D(1024, 3, activation = 'relu', padding = 'same')(base)

    #Build the decoder (expansion path)
    d1 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(base))
    d1 = concatenate([c4, d1], axis = 3)
    d1 = Conv2D(512, 3, activation = 'relu', padding = 'same')(d1)
    d1 = Conv2D(512, 3, activation = 'relu', padding = 'same')(d1)

    d2 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(d1))
    d2 = concatenate([c3, d2], axis = 3)
    d2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(d2)
    d2 = Conv2D(256, 3, activation = 'relu', padding = 'same')(d2)

    d3 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(d2))
    d3 = concatenate([c2, d3], axis = 3)
    d3 = Conv2D(128, 3, activation = 'relu', padding = 'same')(d3)
    d3 = Conv2D(128, 3, activation = 'relu', padding = 'same')(d3)

    d4 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(d3))
    d4 = concatenate([c1, d4], axis = 3)
    d4 = Conv2D(64, 3, activation = 'relu', padding = 'same')(d4)
    d4 = Conv2D(64, 3, activation = 'relu', padding = 'same')(d4)
    d4 = Conv2D(2, 3, padding = 'same')(d4)

    output = Conv2D(2, 1, activation = 'linear')(d4)
    model = Model(inputs, output)
    return model

'''
Training the model using
learning rate schedule with exponential decay,
callbacks, early stopping
'''
def model_train(model, train_x, train_y):

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE_BASE,
        decay_steps=train_x.shape[0]//BATCH_SIZE,
        decay_rate=LEARNING_RATE_DECAY,
        staircase=True)

    #We want the mae as a metric because is robust to outliers
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss='mean_squared_error', metrics='mean_absolute_error')

    if os.path.isfile(modelfile):
        model.load_weights(modelfile)
    checkpoint=ModelCheckpoint(filepath=modelfile, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callback_es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
    csv_logger=CSVLogger(logfile, append=True)

    history = model.fit(train_x, train_y, validation_split=VAL_RATIO, epochs=NEPO, batch_size=BATCH_SIZE, steps_per_epoch=1,
                        callbacks=[checkpoint, callback_es, csv_logger], shuffle=True)

    return history

'''
Calculating training and validation Losses
Following function evaluating model
'''
def convergence_loss(shistory):
    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))
    plt.plot(shistory['loss'], label = "Training Loss")
    plt.plot(shistory['val_loss'], label = "Validation Loss")
    plt.ylim(0,1)
    plt.xlim(0,501)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Loss Convergence', fontsize = 20)
    plt.ylabel('Loss', fontsize = 16)
    plt.xlabel('Epoch', fontsize = 16)
    plt.legend(loc = 'best', fontsize = 16)
    plt.savefig(image_path+'/all_models_final/loss_curve.png')
    #plt.show()

def model_evaluation(model, test_x_norm, test_y_norm):
    #Model evaluation
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_BASE), loss='mean_squared_error', metrics='mean_absolute_error')
    mresult = model.evaluate(test_x_norm, test_y_norm)
    return mresult

'''
Model prediction on the 27 test sample cases
'''
def model_predict(model, test_x_norm, test_y_norm, scale_y_ws, scale_y_ps, test_y, indiciesclw):

    #Model Prediction
    pred_y_norm = model.predict(test_x_norm)
    #Residuals normalized
    residuals_norm = pred_y_norm - test_y_norm

    pred_y_ws = scale_y_ws.inverse_transform(pred_y_norm[:,:,:,0].reshape(-1,1))
    pred_y_ps = scale_y_ps.inverse_transform(pred_y_norm[:,:,:,1].reshape(-1,1))
    pred_y = np.concatenate([np.expand_dims(pred_y_ws.reshape(pred_y_norm[:,:,:,0].shape),axis = -1),
                                np.expand_dims(pred_y_ps.reshape(pred_y_norm[:,:,:,1].shape),axis = -1)], axis=-1)
    #Residuals unnormalized
    residuals = pred_y - test_y
    '''
    ws_norm_rds, ws_rds = residuals_norm[:,:,:,0], residuals[:,:,:,0]
    ps_norm_rds, ps_rds = residuals_norm[:,:,:,1], residuals[:,:,:,1]
    '''
    print("clw shape", np.array(indiciesclw).shape)
    #Summary Stats CLW
    ws_norm_rds, ws_rds = residuals_norm[indiciesclw[0],indiciesclw[1],indiciesclw[2],0], residuals[indiciesclw[0],indiciesclw[1],indiciesclw[2],0]
    ps_norm_rds, ps_rds = residuals_norm[indiciesclw[0],indiciesclw[1],indiciesclw[2],1], residuals[indiciesclw[0],indiciesclw[1],indiciesclw[2],1]
    
    #Summary stats
    ws_norm_stats = summary(ws_norm_rds)
    ps_norm_stats = summary(ps_norm_rds)
    ws_stats = summary(ws_rds)
    ps_stats = summary(ps_rds)
    print("Norm WS", ws_norm_stats)
    print("Norm PS", ps_norm_stats)
    print("UnNorm WS", ws_stats)
    print("UnNorm PS", ps_stats)

    pred_y_ws_flat = pred_y[indiciesclw[0],indiciesclw[1],indiciesclw[2], 0].flatten()
    test_y_ws_flat = test_y[indiciesclw[0],indiciesclw[1],indiciesclw[2], 0].flatten()

    pred_y_ps_flat = pred_y[indiciesclw[0],indiciesclw[1],indiciesclw[2], 1].flatten()
    test_y_ps_flat = test_y[indiciesclw[0],indiciesclw[1],indiciesclw[2],1].flatten()

    # Calculate correlation coefficients
    corr_coeff_ws = pearsonr(pred_y_ws_flat, test_y_ws_flat)[0]
    corr_coeff_ps = pearsonr(pred_y_ps_flat, test_y_ps_flat)[0]

    print("Correlation Coefficient for Wind Speed: ", corr_coeff_ws)
    print("Correlation Coefficient for Surface Pressure: ", corr_coeff_ps)

    return ws_norm_rds, ps_norm_rds, ws_rds, ps_rds, pred_y

'''
Helper function to get residual bias and std stats
'''
def summary(residuals_var):
    residuals_flat = np.reshape(residuals_var, [-1])
    series = pd.Series(residuals_flat)
    summary = series.describe()
    return summary


'''
Histogram plots
'''
def histogram(residuals_var, norm_unorm, wsps, directory):
    # Plot histogram of residuals
    # Fit a normal distribution to the data
    fig, ax = plt.subplots(figsize=(10, 10))
    residuals_flat = np.reshape(residuals_var, [-1])
    mu, std = norm.fit(residuals_flat)
    plt.hist(residuals_flat, bins=30,  edgecolor = 'black', density=True, alpha=0.6, color='blue')
    # Plot the PDF
    x = np.linspace(min(residuals_flat), max(residuals_flat), 30)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, color = 'r', label = r'$\mu,\sigma\ curve$')
    plt.ylabel('Density', fontsize = 30)
    plt.xlim([min(residuals_flat), max(residuals_flat)])
    textstr = '\n'.join((r'$\mathcal{STATISTICS}$', r'$\mu=%.2f$' % (mu, ),
        r'$\sigma=%.3f$' % (std, )))
    props = dict(boxstyle='round', facecolor='grey', alpha=0.5)
    plt.text(0.75, 0.80, textstr, fontsize=30, transform=plt.gca().transAxes, verticalalignment='top', bbox=props)
    if norm_unorm == 0 and wsps == 0:
        plt.title('Normalized WS Residuals')
        plt.xlabel('WS Residuals (m/s)')
        plt.xlim(-2,2)
        plt.savefig(directory + '/HistResidualNormWS.png')
    elif norm_unorm == 0 and wsps == 1:
        plt.title('Normalized Pressure Residuals')
        plt.xlabel('Pressure Residuals (hpa)')
        plt.xlim(-2,2)
        plt.savefig(directory + '/HistResidualNormPS.png')
    elif norm_unorm == 1 and wsps == 0:
        plt.xlabel('WS Residuals (m/s)', fontsize = 30)
        plt.xlim(-10,10)
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.savefig(directory+'/HistResidualUnnormWS.png', bbox_inches='tight')
    elif norm_unorm == 1 and wsps == 1:
        plt.xlabel('Pressure Residuals (hpa)', fontsize = 30)
        plt.xlim(-10,10)
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.savefig(directory+ '/HistResidualUnnormPS.png', bbox_inches='tight')
    plt.legend(loc = 'best')
    #plt.show()

'''
Basemap plots for residuals
'''
def basemap_residual(residuals_var, norm_unorm, wsps, lat , lon, directory):
    """
    input residuals_var is only one record, 2-d
    wsps is flag : 1 pressure, 0 ws
    norm_unorm is flag : 0 is norm, 1 is un norm
    """
    maxlon = np.minimum(np.max(lon) + 5, 180)
    maxlat = np.minimum(np.max(lat) + 5, 90)
    minlon = np.maximum(np.min(lon) - 5, -180)
    minlat = np.maximum(np.min(lat) - 5, -90)

    fig, ax = plt.subplots(figsize=(10, 10))
    m = Basemap(projection='cyl', resolution='l',llcrnrlon=minlon, urcrnrlon=maxlon,llcrnrlat=minlat,urcrnrlat=maxlat)
    x, y = m(lon, lat)
    m.drawcoastlines()
    m.fillcontinents()
    m.drawmapboundary()
    m.drawparallels(np.arange(minlat,maxlat,10.),labels=[1,0,0,0])
    m.drawmeridians(np.arange(minlon,maxlon,20.),labels=[0,0,0,1])
    sc = m.scatter(x, y, c=residuals_var, cmap="rainbow", alpha=0.5)
    cb = m.colorbar(sc, size="5%", pad="2%")
    if norm_unorm == 0 and wsps == 0:
        cb.set_label('WS Residual (m/s)', labelpad=10)
        plt.title('BasemapResidualNormWS')
        sc.set_clim(-2,2)
        plt.savefig(directory + '/BasemapResidualNormWS.png')
    elif norm_unorm == 0 and wsps == 1:
        cb.set_label('Pressure Residual (hpa)', labelpad=10)
        plt.title('BasemapResidualNormPS')
        sc.set_clim(-2,2)
        plt.savefig(directory + '/BasemapResidualNormPS.png')
    elif norm_unorm == 1 and wsps == 0:
        cb.set_label('WS Residual (m/s)', labelpad=10)
        plt.title('Basemap Residual UnNorm Wind Speed')
        sc.set_clim(-5,5)
        plt.savefig(directory+'/BasemapResidualUnnormWS.png')
    elif norm_unorm == 1 and wsps == 1:
        cb.set_label('Pressure Residual (hpa)', labelpad=10)
        sc.set_clim(-10,10)
        plt.title('Basemap Residual UnNorm Pressure')
        plt.savefig(directory+'/BasemapResidualUnnormPS.png')
    #plt.show()

'''
Basemap plots for single sample predictions and labels
'''
def basemap_seperate(arr, wsps, pred_lbl, lat,lon, directory):
    """
    arr is a 2d array
    wsps is flag : 1 pressure, 0 ws
    pred_lbl is flag : 0 is prediction, 1 is label
    """
    '''
    lon = lon.flatten()
    lat = lat.flatten()
    arr = arr.flatten()
    points = np.stack((lon, lat), axis=-1)
    '''
    maxlon = np.round(np.minimum(np.max(lon) + 5, 180),1)
    maxlat = np.round(np.minimum(np.max(lat) + 5, 90),1)
    minlon = np.round(np.maximum(np.min(lon) - 5, -180),1)
    minlat = np.round(np.maximum(np.min(lat) - 5, -90),1)
    fig, ax = plt.subplots(figsize=(10, 10))
    m = Basemap(projection='cyl', resolution='l',llcrnrlon=minlon, urcrnrlon=maxlon,llcrnrlat=minlat,urcrnrlat=maxlat)
    m.drawcoastlines()
    m.fillcontinents()
    m.drawmapboundary()
    m.drawparallels(np.arange(minlat,maxlat,10.),labels=[1,0,0,0], fontsize = 25)
    m.drawmeridians(np.arange(minlon,maxlon,10.),labels=[0,0,0,1], fontsize = 25)
    
    x, y = m(lon, lat)
    sc = m.scatter(x, y, c=arr, cmap="rainbow", alpha=0.5)
    cb = m.colorbar(sc, size="5%", pad="2%")
    cb.ax.tick_params(labelsize=25)
    if pred_lbl == 0 and wsps == 0:
        cb.set_label('AI Prediction WS (m/s)', labelpad=10, fontsize = 30)
        sc.set_clim(0,20)
        cb.mappable.set_clim(0,20)
        plt.savefig(directory+'/BasemapPredWS.png', bbox_inches='tight')
    elif pred_lbl == 0 and wsps == 1:
        cb.set_label('AI Prediction Pressure (hpa)', labelpad=10, fontsize = 30)
        sc.set_clim(960,1030)
        cb.mappable.set_clim(960,1030)
        plt.savefig(directory+'/BasemapPredPS.png', bbox_inches='tight')
    elif pred_lbl == 1 and wsps == 0:
        cb.set_label('ERA5 WS (m/s)', labelpad=10, fontsize = 30)
        sc.set_clim(0,20)
        cb.mappable.set_clim(0,20)
        plt.savefig(directory+'/BasemapLabelWS.png', bbox_inches='tight')
    elif pred_lbl == 1 and wsps == 1:
        cb.set_label('ERA5 Pressure (hpa)', labelpad=10, fontsize = 30)
        sc.set_clim(960,1030)
        cb.mappable.set_clim(960,1030)
        plt.savefig(directory+'/BasemapLabelPS.png', bbox_inches='tight')
    #plt.show()

'''
Scatterplot to compare prediction and test sample images
'''
def scatplot(arr1, arr2, num):
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.scatter(arr1, arr2)
    # y = x line
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')

    # Polynomial best-fit (degree 1 for a straight line)
    coefficients = np.polyfit(arr1, arr2, 1)
    poly = np.poly1d(coefficients)
    sorted_arr1 = np.sort(arr1)
    plt.plot(sorted_arr1, poly(sorted_arr1), color='green', label=f'Best fit')
    R, _ = pearsonr(arr1, arr2)
    if num == 0:
        plt.xlabel('ERA5 WS (m/s)', fontsize = 25)
        plt.ylabel('AI Prediction WS (m/s)', fontsize = 25)
        plt.annotate(f'$R = {R:.2f}$', xy=(10,30), fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.legend(fontsize = 20, loc = 'upper left')
        plt.savefig(image_path +'all_models_final/scatterplotWS.png', bbox_inches='tight')
    elif num == 1:
        plt.xlabel('ERA5 PS (hpa)', fontsize = 25)
        plt.ylabel('AI Prediction PS (hpa)', fontsize = 25)
        plt.annotate(f'$R = {R:.2f}$', xy=(1015.0,1027.5), fontsize=25)
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.legend(fontsize = 20, loc = 'upper left')
        plt.savefig(image_path +'all_models_final/scatterplotPS.png', bbox_inches='tight')

'''
Using system argv to run programs seperately
'''

"""
inp == 1: train
inp == 2: test/predict
inp == 3: plot loss coverage
inp == 4: histogram, map

"""
inp = int(sys.argv[1])

if inp == 1:
    """
    MODEL TRAIN
    """
    print("Start train")
    train_x_norm, train_y_norm, test_x_norm, test_y_norm, train_x, train_y, test_x, test_y, scaler_y_ws, scaler_y_ps, indicies_clw, test_idx  = preprocess(image_path +'hurr_data/*.nc')
    print(train_x_norm.shape, test_x_norm.shape)
    # Train procedure
    mymodel = model_architecture(train_x_norm.shape[1:])
    history = model_train(mymodel, train_x_norm, train_y_norm)
    print("History saving...")
    pickle.dump(history.history, open(hist_file, 'wb'))
    print("Done training")
    
if inp == 2:
    """
    MODEL EVALUATION/PREDICTION
    """
    print("Model Evaluation/Prediction starting")
    train_x_norm, train_y_norm, test_x_norm, test_y_norm, train_x, train_y, test_x, test_y, scaler_y_ws, scaler_y_ps, indicies_clw, test_idx, test_clw  = preprocess(image_path +'hurr_data/*.nc')
    print(indicies_clw)
    files = glob.glob(image_path +'hurr_data/*.nc')
    counter = 1
    for f in files:
        if counter in test_idx:
            print(f)
        counter+=1

    if os.path.isfile(modelfile):
        mymodel = model_architecture(train_x_norm.shape[1:])
        mymodel.load_weights(modelfile)
    else:
        print("mo model file found, exit--")
        exit()

    eval_result = model_evaluation(mymodel, test_x_norm, test_y_norm)
    print("Model Evaluation Results: ", eval_result)
    ws_norm_rds, ps_norm_rds, ws_rds, ps_rds, pred_y  = model_predict(mymodel, test_x_norm, test_y_norm, scaler_y_ws, scaler_y_ps, test_y, indicies_clw)
    np.savez(eval_file, ws_norm_rds=ws_norm_rds, ps_norm_rds=ps_norm_rds, ws_rds=ws_rds, ps_rds=ps_rds, test_x=test_x, test_y=test_y, pred_y=pred_y, test_idx=test_idx)
    print("Done predicting")

if inp == 3:
    if os.path.isfile(hist_file):
        history = pickle.load(open(hist_file, 'rb'))
        curve = convergence_loss(history)
    else:
        print("no history file found. Terminate program.")
        exit()
if inp == 4:
    # Replace with your actual latitude and longitude arrays
    npz=np.load(eval_file)
    test_x=npz["test_x"]
    test_y=npz["test_y"]
    ws_norm_rds=npz["ws_norm_rds"]
    ws_rds=npz["ws_rds"]
    ps_norm_rds=npz["ps_norm_rds"]
    ps_rds=npz["ps_rds"]
    pred_y = npz["pred_y"]
    test_idx = npz["test_idx"]
    test_clw = npz["test_clw"]
    
    maxvals= []
    maxvalsp = []
    maxvalp= []
    maxvalps = []
    for i in range(0,27):
        print('Sample' + str(i) + ' Maps/Histograms')
        directory = image_path + "all_models_final/Sample" + str(i)
        # Create the directory, including all intermediate-level directories, if they don't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        #Basemaps
        latitudes = test_x[i,:,:,0] # example latitudes
        longitudes = test_x[i,:,:,1]  # example longitudes
        #bm_norm_ws = basemap_residual(ws_norm_rds[i], 0,0, latitudes, longitudes, directory)
        #bm_ws = basemap_residual(ws_rds[i], 1,0, latitudes, longitudes, directory)
        #bm_norm_ps = basemap_residual(ps_norm_rds[i], 0,1, latitudes, longitudes, directory)
        #bm_ps = basemap_residual(ps_rds[i], 1,1, latitudes, longitudes, directory)

        #Seperate Basemaps
        lbl_ws = test_y[i,:,:,0]
        lbl_ps = test_y[i,:,:,1]
        pred_y_ws = pred_y[i,:,:,0]
        pred_y_ps = pred_y[i,:,:,1]

        bm_predws = basemap_seperate(pred_y_ws, 0,0, latitudes, longitudes, directory)
        bm_predps = basemap_seperate(pred_y_ps, 1,0, latitudes, longitudes, directory)
        bm_lblws = basemap_seperate(lbl_ws, 0,1, latitudes, longitudes, directory)
        bm_lblps = basemap_seperate(lbl_ps, 1,1, latitudes, longitudes, directory)

        #Histograms
        #wshistnorm = histogram(ws_norm_rds[i], 0, 0, directory)
        #pshistnorm = histogram(ps_norm_rds[i], 0, 1, directory)
        wshist = histogram(ws_rds[i], 1, 0, directory)
        pshist = histogram(ps_rds[i], 1, 1, directory)

        maxvals.append(np.max(lbl_ws))
        maxvalsp.append(np.max(pred_y_ws))
        minvalp.append(np.min(lbl_ps))
        minvalps.append(np.min(pred_y_ps))

    np.savez(image_path +'all_models_final/maxvalues.dat.npz', maxvals=maxvals, maxvalsp=maxvalsp, minvalp=minvalp, minvalps=minvalps)

if inp == 5:
    npz=np.load(image_path +'all_models_final/maxvalues.dat.npz')
    maxvals=npz["maxvals"]
    maxvalsp=npz["maxvalsp"]
    maxvalp=npz["maxvalp"]
    maxvalps=npz["maxvalps"]
    scatws = scatplot(maxvals, maxvalsp, 0)
    scatps = scatplot(maxvalp, maxvalps, 1)
if inp == 6:
    train_x_norm, train_y_norm, test_x_norm, test_y_norm, train_x, train_y, test_x, test_y, scaler_y_ws, scaler_y_ps, indicies_clw, test_idx  = preprocess(image_path +'hurr_data/*.nc')    
    print(np.max(train_y[:,:,:,0]),np.max(train_y[:,:,:,1]),np.min(train_y[:,:,:,0]), np.min(train_y[:,:,:,1]))
