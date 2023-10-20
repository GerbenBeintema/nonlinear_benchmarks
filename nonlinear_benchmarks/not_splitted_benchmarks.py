

import os
from scipy.io import loadmat
import tempfile
import os.path
from pathlib import Path
from nonlinear_benchmarks.utilities import *
import numpy as np


def BoucWen(train_test_split=True, raw_data=False, dir_placement=None, force_download=False, url=None):

    #todo: dot p file integration as system for training data
    #generate more data
    # url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/BOUCWEN/BoucWenFiles.zip'
    url = 'https://data.4tu.nl/ndownloader/files/24703124'
    download_size = 5284363
    save_dir = cashed_download(url,'BoucWen',zip_name='BoucWenFiles.zip',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'BoucWenFiles/Test signals/Validation signals') #matfiles location
    if raw_data:
        return save_dir

    if train_test_split==True:
        print('Warning no offical train and test split has been determined for this dataset')
    
    datafiles = []

    out = loadmat(os.path.join(save_dir,'uval_multisine.mat'))
    u_multisine = out['uval_multisine'][0]
    out = loadmat(os.path.join(save_dir,'yval_multisine.mat'))
    y_multisine = out['yval_multisine'][0]
    datafiles.append(Input_output_data(u=u_multisine,y=y_multisine))

    out = loadmat(os.path.join(save_dir,'uval_sinesweep.mat'))
    u_sinesweep = out['uval_sinesweep'][0]
    out = loadmat(os.path.join(save_dir,'yval_sinesweep.mat'))
    y_sinesweep = out['yval_sinesweep'][0]
    datafiles.append(Input_output_data(u=u_sinesweep,y=y_sinesweep))
    return datafiles


def WienerHammerstein_Process_Noise(train_test_split=True, raw_data=False, dir_placement=None, force_download=False, url=None):
    '''Warning this is a quite a bit of data'''
    # url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/WIENERHAMMERSTEINPROCESS/WienerHammersteinFiles.zip'
    url = 'https://data.4tu.nl/ndownloader/files/24671987'
    download_size=423134764
    save_dir = cashed_download(url,'WienHammer',zip_name='WienerHammersteinFiles.zip',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'WienerHammersteinFiles') #matfiles location
    matfiles = [os.path.join(save_dir,a).replace('\\','/') for a in os.listdir(save_dir) if a.split('.')[-1]=='mat']

    if train_test_split==True:
        print('Warning no offical train and test split has been determined for this dataset')

    if raw_data:
        return matfiles
    
    dataset = []
    dataset_test = []

    #file = 'WH_CombinedZeroMultisineSinesweep.mat' #'WH_Triangle_meas.mat'
    for file in matfiles:
        out = loadmat(os.path.join(save_dir,file))
        r,u,y,fs = out['dataMeas'][0,0]
        fs = fs[0,0]
        data = [Input_output_data(u=ui,y=yi, sampling_time=1/fs) for ui,yi in zip(u.T,y.T)]
        if train_test_split and 'Test' in file:
            dataset_test.extend(data)
        else:
            dataset.extend(data)
    return dataset, dataset_test



def ParWHF(train_test_split=True, raw_data=False, dir_placement=None, force_download=False, url=None):
    '''Parallel Wienner-Hammerstein'''
    # url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/PARWH/ParWHFiles.zip'
    url = 'https://data.4tu.nl/ndownloader/files/24666227' if url is None else url
    download_size=58203304
    save_dir = cashed_download(url,'ParWHF',zip_name='ParWHFiles.zip',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'ParWHFiles') #matfiles location
    d = os.path.join(save_dir,'ParWHData.mat')
    if raw_data:
        return d

    out = loadmat(d)
    # print(out.keys())
    # print(out['amp'][0]) #5 values 
    # print(out['fs'][0,0])
    # print(out['lines'][0]) #range 2:4096
    # print('uEst',out['uEst'].shape) #(16384, 2, 20, 5), (N samplees, P periods, M Phase changes, nAmp changes)
    # print('uVal',out['uVal'].shape) #(16384, 2, 1, 5)
    # print('uValArr',out['uValArr'].shape) #(16384, 2)
    # print('yEst',out['yEst'].shape) #(16384, 2, 20, 5)
    # print('yVal',out['yVal'].shape) #(16384, 2, 1, 5)
    # print('yValArr',out['yValArr'].shape) #(16384, 2)
    if train_test_split==True:
        print('Warning no offical train and test split has been determined for this dataset')

    fs = out['fs'][0,0]

    datafiles = []
    datafiles_test = []
    #todo split train, validation and test
    uEst = out['uEst'].reshape((16384,-1))
    yEst = out['yEst'].reshape((16384,-1))
    datafiles.extend([Input_output_data(u=ui,y=yi, sampling_time=1/fs) for ui,yi in zip(uEst.T,yEst.T)])
    
    uVal = out['uVal'].reshape((16384,-1))
    yVal = out['yVal'].reshape((16384,-1))
    data = [Input_output_data(u=ui,y=yi, sampling_time=1/fs) for ui,yi in zip(uVal.T,yVal.T)]
    datafiles_test.extend(data) if train_test_split else datafiles.extend(data)
    
    uValArr = out['uValArr'].reshape((16384,-1))
    yValArr = out['yValArr'].reshape((16384,-1))
    data = [Input_output_data(u=ui,y=yi, sampling_time=1/fs) for ui,yi in zip(uValArr.T,yValArr.T)]
    datafiles_test.extend(data) if train_test_split else datafiles.extend(data)

    return datafiles, datafiles_test

    
def F16(train_test_split=True, output_index=0, raw_data=False, dir_placement=None, force_download=False, url=None):
    '''The F-16 Ground Vibration Test benchmark features a high order system with clearance and friction nonlinearities at the mounting interface of the payloads.

    The experimental data made available to the Workshop participants were acquired on a full-scale F-16 aircraft on the occasion of the Siemens LMS Ground Vibration Testing Master Class, held in September 2014 at the Saffraanberg military basis, Sint-Truiden, Belgium.

    During the test campaign, two dummy payloads were mounted at the wing tips to simulate the mass and inertia properties of real devices typically equipping an F-16 in ﬂight. The aircraft structure was instrumented with accelerometers. One shaker was attached underneath the right wing to apply input signals. The dominant source of nonlinearity in the structural dynamics was expected to originate from the mounting interfaces of the two payloads. These interfaces consist of T-shaped connecting elements on the payload side, slid through a rail attached to the wing side. A preliminary investigation showed that the back connection of the right-wing-to-payload interface was the predominant source of nonlinear distortions in the aircraft dynamics, and is therefore the focus of this benchmark study.

    A detailed formulation of the identification problem can be found here. All the provided files and information on the F-16 aircraft benchmark system are available for download here. This zip-file contains a detailed system description, the estimation and test data sets, and some pictures of the setup. The data is available in the .csv and .mat file format.

    Please refer to the F16 benchmark as:

    J.P. Noël and M. Schoukens, F-16 aircraft benchmark based on ground vibration test data, 2017 Workshop on Nonlinear System Identification Benchmarks, pp. 19-23, Brussels, Belgium, April 24-26, 2017.

    Previously published results on the F-16 Ground Vibration Test benchmark are listed in the history section of this webpage.

    Special thanks to Bart Peeters (Siemens Industry Software) for his help in creating this benchmark.'''
    #todo this is still broken for some mat files
    # assert False, 'this is still broken for some files where y has many more dimensions than expected'
    # url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/F16/F16GVT_Files.zip'

    if train_test_split==True:
        print('Warning no offical train and test split has been determined for this dataset')

    url = 'https://data.4tu.nl/ndownloader/files/24675560' if url is None else url
    download_size=148455295
    save_dir = cashed_download(url,'F16',zip_name='F16GVT_Files.zip',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'F16GVT_Files/BenchmarkData') #matfiles location
    matfiles = [os.path.join(save_dir,a).replace('\\','/') for a in os.listdir(save_dir) if a.split('.')[-1]=='mat']
    if raw_data:
        return matfiles
    datasets = []
    for file in sorted(matfiles):
        out = loadmat(file)
        Force, Voltage, (y1,y2,y3),Fs = out['Force'][0], out['Voltage'][0], out['Acceleration'], out['Fs'][0,0]
        #u = Force
        #y = one of the ys, multi objective regression?
        name = file.split('/')[-1]
        if 'SpecialOddMSine' not in name:
            datasets.append(Input_output_data(u=Force,y=[y1,y2,y3][output_index], sampling_time=1/Fs))
    return datasets


def Industrial_robot(train_test_split=True, raw_data=False, dir_placement=None, force_download=False, url=None):
    '''An identification benchmark dataset for a full robot movement with a KUKA KR300 R2500 
    ultra SE industrial robot is presented. It is a robot with a nominal payload capacity of
    300 kg, a weight of 1120 kg, and a reach of 2500mm. It exhibits 12 states accounting for
    position and velocity for each of the 6 joints. The robot encounters backlash in all 
    joints, pose-dependent inertia, pose-dependent gravitational loads, pose-dependent
    hydraulic forces, pose- and velocity-dependent centripetal and Coriolis forces as well 
    as nonlinear friction, which is temperature-dependent and therefore potentially 
    time-varying. Prepared datasets for black-box identification of the forward or the 
    inverse robot dynamics are provided. Additional to the data for the black-box modeling, 
    we supply high-frequency raw data and videos of each experiment. A baseline and figures
    of merit are defined to make results comparable across different identification methods.

    A detailed formulation of the identification problem can be found here. All the provided 
    files and information on the industrial robot dataset can be found here. 

    https://kluedo.ub.uni-kl.de/frontdoor/index/index/docId/6731

    https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/

    https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/Robot_Identification_Benchmark_Without_Raw_Data.rar

    Special thanks to Jonas Weigand and co-authors for creating and sharing this benchmark!'''
    url = 'https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/Robot_Identification_Benchmark_Without_Raw_Data.rar'  if url is None else url




    download_size=12717003 
    save_dir = cashed_download(url, 'Industrial_robot', zip_name='Robot_Identification_Benchmark_Without_Raw_Data.rar',\
        dir_placement=dir_placement, download_size=download_size, force_download=force_download)
    # save_dir = os.path.join(save_dir,'forward_identification_without_raw_data') #matfiles location
    d = os.path.join(save_dir,'forward_identification_without_raw_data.mat')
    if raw_data:
        return d


    if train_test_split==True:
        print('Warning no offical train and test split has been determined for this dataset')


    out = loadmat(d)

    K = 606 
    trains = [Input_output_data(y=out['y_train'][:,n*K:(n+1)*K].T,u = out['u_train'][:,n*K:(n+1)*K].T) \
            for n in range(out['y_train'].shape[1]//K)]
    train = trains
    tests = [Input_output_data(y=out['y_test'][:,n*K:(n+1)*K].T,u = out['u_test'][:,n*K:(n+1)*K].T) \
            for n in range(out['y_test'].shape[1]//K)]
    test = tests

    return train, test
