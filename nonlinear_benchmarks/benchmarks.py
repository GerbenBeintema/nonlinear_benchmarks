


# http://www.nonlinearbenchmark.org/
# A. Janot, M. Gautier and M. Brunot, Data Set and Reference Models of EMPS, 2019 Workshop on Nonlinear System Identification Benchmarks, Eindhoven, The Netherlands, April 10-12, 2019.

import os
from scipy.io import loadmat
import tempfile
import os.path
from pathlib import Path
from nonlinear_benchmarks.utilities import *
import numpy as np

def EMPS(train_test_split=True, data_file_locations=False, dir_placement=None, force_download=False, \
         url=None, atleast_2d=False, always_return_tuples_of_datasets=False):
    '''The Electro-Mechanical Positioning System is a standard configuration of a drive system for prismatic joint of robots or machine tools. The main source of nonlinearity is caused by friction effects that are present in the setup. Due to the presence of a pure integrator in the system, the measurements are obtained in a closed-loop setting.

    The provided data is described in this link. The provided Electro-Mechanical Positioning System datasets are available for download here. This zip-file contains the system description and available data sets .mat file format.

    Please refer to the Electro-Mechanical Positioning System as:

    A. Janot, M. Gautier and M. Brunot, Data Set and Reference Models of EMPS, 2019 Workshop on Nonlinear System Identification Benchmarks, Eindhoven, The Netherlands, April 10-12, 2019.

    Special thanks to Alexandre Janot for making this dataset available.'''
    #q_cur current measured position
    #q_ref target/reference potion
    #non-linear due to singed friction force Fc ~ sing(dq/dt)
    #t time
    #vir applied the vector of motor force expressed in the load side i.e. in N;

    # url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/EMPS/EMPS.zip'
    url = 'https://drive.google.com/file/d/1zwoXYa9-3f8NQ0ohzmjpF7UxbNgRTHkS/view' if url is None else url
    download_size = 1949929
    save_dir = cashed_download(url,'EMPS',zip_name='EMPS.zip', dir_placement=dir_placement,download_size=download_size,force_download=force_download)

    if data_file_locations:
        return [os.path.join(save_dir,name) for name in ['DATA_EMPS.mat','DATA_EMPS_PULSES.mat']]

    datasets = []
    for name, file_name in [('train (DATA_EMPS)','DATA_EMPS.mat'),('test (DATA_EMPS_PULSES)','DATA_EMPS_PULSES.mat')]:
        matfile = loadmat(os.path.join(save_dir,file_name))
        q_cur, q_ref, t, vir = [matfile[a][:,0] for a in ['qm','qg','t','vir']] #qg is reference, either, q_ref is input or vir is input
        out_data = Input_output_data(u=vir, y=q_cur, sampling_time=t[1]-t[0], name=name)
        datasets.append(out_data)




    if train_test_split:
        train_val = datasets[0]
        test = datasets[1]
        test.state_initialization_window_length = 20
        return always_return_tuples_of_datasets_fun(*atleast_2d_fun(train_val, test, apply=atleast_2d), apply=always_return_tuples_of_datasets)
    else:
        return atleast_2d_fun(datasets, apply=atleast_2d)

def CED(train_test_split=True, data_file_locations=False, dir_placement=None, force_download=False, \
        url=None, atleast_2d=False, always_return_tuples_of_datasets=False):
    '''The coupled electric drives consists of two electric motors that drive a pulley using a flexible belt. 
    The pulley is held by a spring, resulting in a lightly damped dynamic mode. The electric drives can
    be individually controlled allowing the tension and the speed of the belt to be simultaneously controlled. 
    The drive control is symmetric around zero, hence both clockwise and counter clockwise movement is possible.
    The focus is only on the speed control system. The angular speed of the pulley is measured as an output with
    a pulse counter and this sensor is insensitive to the sign of the velocity. The available data sets are short,
    which constitute a challenge when performing identification.

    The provided data is part of a technical note available online through this link. 
    The provided Coupled Electric Drives datasets are available for download here. 
    This zip-file contains the system description and available data sets, both in 
    the .csv and .mat file format.

    Please refer to the Coupled Electric Drives dataset as:

    T. Wigren and M. Schoukens, Coupled Electric Drives Data Set and Reference Models, 
    Technical Report, Department of Information Technology, Uppsala University, 
    Department of Information Technology, Uppsala University, 2017.

    Previously published results on the Coupled Electric Drives benchmark are listed in 
    the history section of this webpage.

    Special thanks to Torbjön Wigren for making this dataset available.

    NOTE: We are re-evaluating the continuous-time models reported in the technical note. 
    For now, the discrete-time model reported in eq. (9) of the technical note can be used 
    in combination of PRBS dataset with amplitude 1.'''

    #http://www.it.uu.se/research/publications/reports/2017-024/2017-024-nc.pdf
    url = 'http://www.it.uu.se/research/publications/reports/2017-024/CoupledElectricDrivesDataSetAndReferenceModels.zip' if url is None else url
    download_size= 278528
    save_dir = cashed_download(url,'CED',dir_placement=dir_placement,download_size=download_size,force_download=force_download)

    d = os.path.join(save_dir,'DATAUNIF.MAT')
    if data_file_locations:
        return d

    matfile = loadmat(d)
    u11, u12, z11, z12 = [matfile[a][:,0] for a in ['u11','u12','z11','z12']]
    datasets = [Input_output_data(u=u11, y=z11, sampling_time=0.02, name='CED low input amplitude'), \
                Input_output_data(u=u12,y=z12, sampling_time=0.02, name='CED high input amplitude')]

    if train_test_split:
        train_val = datasets[0][:400], datasets[1][:400]
        train_val[0].name = 'train CED low input amplitude'
        train_val[1].name = 'train CED high input amplitude'
        test = datasets[0][400:], datasets[1][400:]
        test[0].name = 'test CED low input amplitude'
        test[1].name = 'test CED high input amplitude'
        test[0].state_initialization_window_length = 4
        test[1].state_initialization_window_length = 4
        return atleast_2d_fun(train_val, test, apply=atleast_2d)
    else:
        return atleast_2d_fun(datasets, apply=atleast_2d)

def Cascaded_Tanks(train_test_split=True, data_file_locations=False, dir_placement=None, force_download=False, url=None, \
    atleast_2d=False, always_return_tuples_of_datasets=False):
    #does not work anymore?
    # url = 'https://data.4tu.nl/file/d4810b78-6cdd-48fe-8950-9bd601e5f47f/3b697e42-01a4-4979-a370-813a456c36f5' if url is None else url
    url = 'https://drive.google.com/file/d/1HnQf_gu0g_UlggoBqy2s34l9YJiFdN01/view' if url is None else url
    download_size = 7520592
    save_dir = cashed_download(url, 'Cascaded_Tanks', zip_name='CascadedTanksFiles.zip',dir_placement=dir_placement,download_size=download_size,force_download=force_download)
    save_dir = os.path.join(save_dir,'CascadedTanksFiles')

    d = os.path.join(save_dir,'dataBenchmark.mat')
    if data_file_locations:
        return d
    out = loadmat(d)

    uEst, uVal, yEst, yVal, Ts = out['uEst'][:,0],out['uVal'][:,0],out['yEst'][:,0],out['yVal'][:,0],out['Ts'][0,0]
    datasets = [Input_output_data(u=uEst,y=yEst, sampling_time=Ts, name='train Cascaded_Tanks'),\
                Input_output_data(u=uVal,y=yVal, sampling_time=Ts, name='test Cascaded_Tanks')]
    if train_test_split:
        train_val = datasets[0]
        test = datasets[1]
        test.state_initialization_window_length = 5
        return always_return_tuples_of_datasets_fun(*atleast_2d_fun(train_val, test, apply=atleast_2d), apply=always_return_tuples_of_datasets)
    else:
        datasets[0].name, datasets[1].name = 'Cascaded_Tanks first dataset', 'Cascaded_Tanks second dataset'
        return atleast_2d_fun(datasets, apply=atleast_2d)

def WienerHammerBenchMark(train_test_split=True, data_file_locations=False, dir_placement=None, force_download=False, url=None, \
    atleast_2d=False, always_return_tuples_of_datasets=False):
    url = 'http://www.ee.kth.se/~hjalmars/ifac_tc11_benchmarks/2009_wienerhammerstein/WienerHammerBenchMark.mat' if url is None else url
    download_size=1707601
    save_dir = cashed_download(url,'WienerHammerBenchMark',dir_placement=dir_placement,download_size=download_size,force_download=force_download,zipped=False)

    if data_file_locations:
        return os.path.join(save_dir,'WienerHammerBenchMark.mat')

    out = loadmat(os.path.join(save_dir,'WienerHammerBenchMark.mat'))
    u,y,fs = out['uBenchMark'][:,0], out['yBenchMark'][:,0], out['fs'][0,0]
    full_data = Input_output_data(u=u,y=y, sampling_time=1/fs, name='WH benchmark full')
    if train_test_split==False:
        V = atleast_2d_fun(full_data, apply=atleast_2d)
        return (V,) if always_return_tuples_of_datasets else V
    
    sys_data = full_data[5200:184000] 
    train, test = sys_data[:100000], sys_data[100000:]
    train.name = 'train WH'
    test.name = 'test WH'
    test.state_initialization_window_length = 50
    return always_return_tuples_of_datasets_fun(*atleast_2d_fun(train, test, apply=atleast_2d), apply=always_return_tuples_of_datasets)

def Silverbox(train_test_split=True, data_file_locations=False, dir_placement=None, force_download=False, url=None, \
    atleast_2d=False, always_return_tuples_of_datasets=False):
    '''The Silverbox system can be seen as an electronic implementation of the Duffing oscillator. It is build as a 
    2nd order linear time-invariant system with a 3rd degree polynomial static nonlinearity around it in feedback. 
    This type of dynamics are, for instance, often encountered in mechanical systems.

    The provided data is part of a previously published ECC paper available online. A technical note describing the 
    Silverbox benchmark can be found here. All the provided data (.mat file format) on the Silverbox system is available
    for download here. This .zip file contains the Silverbox dataset as specified in the benchmark document (V1 is the
    input record, while V2 is the measured output), extended with .csv version of the same data and an extra data record 
    containing a Schroeder phase multisine measurement.

    Please refer to the Silverbox benchmark as:

    T. Wigren and J. Schoukens. Three free data sets for development and benchmarking in nonlinear system identification. 
    2013 European Control Conference (ECC), pp.2933-2938 July 17-19, 2013, Zurich, Switzerland.

    Previously published results on the Silverbox benchmark are listed in the history section of this webpage.

    Special thanks to Johan Schoukens for creating this benchmark, and to Torbjörn Wigren for hosting this benchmark.
    '''
    # url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/SILVERBOX/SilverboxFiles.zip' #old
    url = 'https://drive.google.com/file/d/17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy/view' if url is None else url
    download_size=5793999
    save_dir = cashed_download(url, 'Silverbox', zip_name='SilverboxFiles.zip',\
        dir_placement=dir_placement, download_size=download_size, force_download=force_download)
    save_dir = os.path.join(save_dir,'SilverboxFiles') #matfiles location

    d1, d2 = os.path.join(save_dir,'Schroeder80mV.mat'), os.path.join(save_dir,'SNLS80mV.mat')
    if data_file_locations:
        return d1, d2

    out = loadmat(d2) #train test
    u,y = out['V1'][0], out['V2'][0]
    data2 = Input_output_data(u=u,y=y, sampling_time=1/610.35, name='Silverbox Data All')

    if train_test_split:
        test_arrow_full = data2[100:40575]
        test_arrow_full.name = 'test SB multisine'
        test_arrow_no_extrapolation = test_arrow_full[:32000] #keep init=50
        test_arrow_no_extrapolation.name = 'test SB arrow no extrapolation'

        multisine = data2[40650:127400]
        s = int(len(multisine)*0.75)
        multisine_train_val, test_multisine = multisine[:s], multisine[s:]
        multisine_train_val.name = 'train SB multisine'
        test_multisine.name = 'test SB multisine'
        for v in [test_multisine, test_arrow_full, test_arrow_no_extrapolation]:
            v.state_initialization_window_length = 50

        from collections import namedtuple
        # m = namedtuple('Silverbox_data_splits', ['test_multisine', 'test_arrow_full', 'test_arrow_no_extrapolation'])
        multisine_train_val = (multisine_train_val,) if always_return_tuples_of_datasets else multisine_train_val
        return atleast_2d_fun(multisine_train_val, (test_multisine, test_arrow_full, test_arrow_no_extrapolation), apply=atleast_2d)
    else:
        data2 = (data2,) if always_return_tuples_of_datasets else data2
        return atleast_2d_fun(data2, apply=atleast_2d)

