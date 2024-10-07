

import os
from scipy.io import loadmat
import tempfile
import os.path
from pathlib import Path
from nonlinear_benchmarks.utilities import *
import numpy as np


def BoucWen(train_test_split=True, data_file_locations=False, dir_placement=None, force_download=False, url=None):

    #todo: dot p file integration as system for training data
    #generate more data
    # url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/BOUCWEN/BoucWenFiles.zip'
    # url = 'https://data.4tu.nl/ndownloader/items/7060f9bc-8289-411e-8d32-57bef2740d32/versions/1'
    url = 'https://data.4tu.nl/file/7060f9bc-8289-411e-8d32-57bef2740d32/cd40469c-5064-4968-ae59-88cbb850264b' if url is None else url
    download_size = 5649141
    save_dir = cashed_download(url,'BoucWen',zip_name='Hysteretic Benchmark with a Dynamic Nonlinearity_1_all.zip',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'BoucWenFiles/Test signals/Validation signals') #matfiles location
    if data_file_locations:
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


def WienerHammerstein_Process_Noise(train_test_split=True, data_file_locations=False, dir_placement=None, force_download=False, url=None):
    '''Warning this is a quite a bit of data'''
    # url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/WIENERHAMMERSTEINPROCESS/WienerHammersteinFiles.zip'
    url = 'https://data.4tu.nl/file/1f194001-affa-4459-870a-ad9e9d9146f9/2dbbc046-1ac2-43b2-bf4e-53b5a4be8b96' if url is None else url
    download_size=423134764
    save_dir = cashed_download(url,'WienHammer',zip_name='WienerHammersteinFiles.zip',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'WienerHammersteinFiles') #matfiles location
    matfiles = [os.path.join(save_dir,a).replace('\\','/') for a in os.listdir(save_dir) if a.split('.')[-1]=='mat']

    if train_test_split==True:
        print('Warning no offical train and test split has been determined for this dataset')

    if data_file_locations:
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
    
def Industrial_robot(train_test_split=True, data_file_locations=False, dir_placement=None, force_download=False, url=None):
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

    https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/Robot_Identification_Benchmark_Without_data_file_locations.rar

    Special thanks to Jonas Weigand and co-authors for creating and sharing this benchmark!'''
    url = 'https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/Robot_Identification_Benchmark_Without_Raw_Data.rar'  if url is None else url




    download_size = 12717003 
    save_dir = cashed_download(url, 'Industrial_robot', zip_name='Robot_Identification_Benchmark_Without_Raw_Data.rar',\
        dir_placement=dir_placement, download_size=download_size, force_download=force_download)
    # save_dir = os.path.join(save_dir,'forward_identification_without_data_file_locations') #matfiles location
    d = os.path.join(save_dir,'forward_identification_without_raw_data.mat')
    if data_file_locations:
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

if __name__=='__main__':
    data = F16()
