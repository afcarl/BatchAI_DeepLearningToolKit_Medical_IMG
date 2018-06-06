# Databricks notebook source
## Read the IXI Dataset from the datalakestore (use datafactory copy http tool to copy data at IXIDataset site)
#
# https://docs.microsoft.com/en-us/azure/data-lake-store/data-lake-store-data-operations-python
# Note: pip install azure-mgmt-resource, azure-mgmt-datalake-store, azure-datalake-store [azure.datalake.store]
#
## Use this only for Azure AD service-to-service authentication
from azure.common.credentials import ServicePrincipalCredentials

## Use this only for Azure AD end-user authentication
from azure.common.credentials import UserPassCredentials

## Use this only for Azure AD multi-factor authentication
from msrestazure.azure_active_directory import AADTokenCredentials

## Required for Azure Data Lake Store account management
from azure.mgmt.datalake.store import DataLakeStoreAccountManagementClient
from azure.mgmt.datalake.store.models import DataLakeStoreAccount

## Required for Azure Data Lake Store filesystem management
from azure.datalake.store import core, lib, multithread

# Common Azure imports
from azure.mgmt.resource.resources import ResourceManagementClient
from azure.mgmt.resource.resources.models import ResourceGroup

## Use these as needed for your application
import logging, getpass, pprint, uuid, time

##Note: if any lib iumport not found, likely is the pkg index outddated, goto https://pypi.org/ to search the lib and specified the version in Databricks lib install

# COMMAND ----------

##--Option1(SignIn): User Acct, interactively sign in to AzDataLake Store, storebotdatalakestore
## 
##subscriptionId = '851da8fc-5b5f-48f2-9e14-395ce8ace4bf'
adlsAccountName = 'storebotdatalakestore'

adlCreds = lib.auth(tenant_id = '4dced229-4c95-476d-b76b-34d306d723eb', resource = 'https://datalake.azure.net/')
##--Create a filesystem client object
adlsFileSystemClient = core.AzureDLFileSystem(adlCreds, store_name=adlsAccountName)

# COMMAND ----------

##--Option2(SignIn): Authenticate non-interactively to AzDataLake Store, storebotdatalakestore
## NOT YET WORK, Got Permission in below access!!
##
subscriptionId = '851da8fc-5b5f-48f2-9e14-395ce8ace4bf'
adlsAccountName = 'storebotdatalakestore'
databricksADApp_client_id = '55f2dd64-39d7-4fa5-a12b-5da3bf6e1d50'
databricksADApp_client_secret = 'yeZZZBYVlWvdbAhPA4EulPtNrlai0GZwdb6vOoHwl8Q='
RESOURCE = 'https://datalake.azure.net/'
authority_host_uri = 'https://login.microsoftonline.com'
tenant = '4dced229-4c95-476d-b76b-34d306d723eb'
authority_uri = authority_host_uri + '/' + tenant + '/oauth2/token'

##import adal
##context = adal.AuthenticationContext(authority_uri, api_version=None)
##mgmt_token = context.acquire_token_with_client_credentials(RESOURCE, databricksADApp_client_id, databricksADApp_client_secret)
##armCreds = AADTokenCredentials(mgmt_token, databricksADApp_client_id, resource=RESOURCE)

adlCreds = lib.auth(tenant_id=tenant, client_secret=databricksADApp_client_secret, client_id=databricksADApp_client_id, resource=RESOURCE)
adlsFileSystemClient = core.AzureDLFileSystem(adlCreds, store_name=adlsAccountName)

##--Don't RUN this for now, as it seems there's some permission access issue after Auth successfully????????

# COMMAND ----------

# Pandas data frame read directly from Datalaks Store's file system
import pandas as pd

with adlsFileSystemClient.open('/clusters/DLTK_IXI_Dataset/bvals.txt', 'rb') as f:
    df = pd.read_csv(f) 
# Show the dataframe
df  

# COMMAND ----------

with adlsFileSystemClient.open('/clusters/DLTK_IXI_Dataset/IXI.xls', 'rb') as f:
    xls= pd.ExcelFile(f)
# Show the dataframe's sheet name only
print(xls.sheet_names)

# COMMAND ----------

##import tarfile
#--TESTING only!!!!!
#--Get the tar files from datalakestore
#--
#--Ref: http://azure-datalake-store.readthedocs.io/en/latest/api.html
##t1_tar = adlsFileSystemClient.cat('/clusters/DLTK_IXI_Dataset/IXI-T1.tar')
#--Open tarfile
##tar = tarfile.open(t1_tar, 'r')

#--Iterate over every member
##for tarinfo in tar:
##    print (tarinfo.name, "is", tarinfo.size, "bytes in size and is"),
##    if tarinfo.isreg():
##        print ("a regular file.")
##    elif tarinfo.isdir():
##        print ("a directory.")
##    else:
##        print ("something else.")
##    tar.close()  

# COMMAND ----------

# NO NEED TO DOWNLOAD THE TAR FILES FROM datalakestore to the datalake File System, JUST Directly READ FROM datalakestore
# Note: 
#  - somehow the donwlod files are not shown in the local folder if run on Databricks, DLTK_IXI_Dataset?????
#  - it's too slow so lets try mutlithread download
#  - File will stored at the session directory C:\Users\wdam\AppData\Local\Temp\azureml_runs\tensorflow-tutorial_xxx E.g. tensorflow-tutorial_1520827044285
#
# Get the tar files from datalakestore
##adlsFileSystemClient.get('/clusters/DLTK_IXI_Dataset/IXI-T1.tar', 'DLTK_IXI_Dataset/IXI_T1.tar')
##adlsFileSystemClient.get('/clusters/DLTK_IXI_Dataset/IXI-T2.tar', 'DLTK_IXI_Dataset/IXI_T2.tar')
##adlsFileSystemClient.get('/clusters/DLTK_IXI_Dataset/IXI-MRA.tar', 'IXI_MRA.tar')
##adlsFileSystemClient.get('/clusters/DLTK_IXI_Dataset/IXI-PD.tar', 'IXI_PD.tar')
##adlsFileSystemClient.get('/clusters/DLTK_IXI_Dataset/IXI.xls', 'demographic.xls')

# COMMAND ----------

# Create a dolder DLTK_IXI_Dataset in the root datalakestore, which is under storebotdatlakestore
##adlsFileSystemClient.mkdir(path='DLTK_IXI_Dataset')

# COMMAND ----------

# NO NEED TO DOWNLOAD THE TAR FILES, JUST Directly READ FROM datalakestore
# Note: somehow the donwlod files are not shown in the local folder if run on Databricks, DLTK_IXI_Dataset?????
# recursively download the whole directory tree with 10 threads and
# 16MB chunks
# multithread.ADLDownloader(adlsFileSystemClient, "/clusters/DLTK_IXI_Dataset", 'DLTK_IXI_Dataset', 10, 2**24, overwrite=True)

# COMMAND ----------

# List the files under the datalakestore (previously using Datafactory (as binary) to copy files into datalakestore)
adlsFileSystemClient.ls('/clusters/DLTK_IXI_Dataset')

# COMMAND ----------

# Checking one of the downlad files(tar)
import tarfile
with adlsFileSystemClient.open('/clusters/DLTK_IXI_Dataset/IXI-T1.tar', 'rb') as f:
  # Open tarfile
  IXI_T1_tar = tarfile.open(name='IXI-T1.tar', fileobj=f, mode='r', debug=2)
  # Iterate over every member
  for tarinfo in IXI_T1_tar:
    print (tarinfo.name, "is", tarinfo.size, "bytes in size and is"),
    if tarinfo.isreg():
        print ("a regular file.")
    elif tarinfo.isdir():
        print ("a directory.")
    else:
        print ("something else.")
  ##tar.close()

# COMMAND ----------

##Ref: https://github.com/DLTK/DLTK/blob/master/data/IXI_HH/download_IXI_HH.py
#      install "SimpleITK> 1.0.1" instaed of "SimpleITK" to the databricks library
import os.path
import glob
import SimpleITK as sitk
import numpy as np

EXTRACT_IMAGES = True
PROCESS_OTHER = True
RESAMPLE_IMAGES = True
CLEAN_UP = True

# COMMAND ----------

# show the local file path
##os.listdir("/clusters/DLTK_IXI_Dataset")

# COMMAND ----------

# show the local file path
##os.listdir("/clusters/DLTK_IXI_Dataset/T2")

# COMMAND ----------

# Create datlakestore directories for below IXI_Dataset tar file extraction
# Note: Since its need to done once, so comment it
#adlsFileSystemClient.mkdir(path='/clusters/DLTK_IXI_Dataset/T1')
##adlsFileSystemClient.mkdir(path='/clusters/DLTK_IXI_Dataset/T2')
##adlsFileSystemClient.mkdir(path='/clusters/DLTK_IXI_Dataset/MRA')
##adlsFileSystemClient.mkdir(path='/clusters/DLTK_IXI_Dataset/PD')
#
##adlsFileSystemClient.mkdir('/clusters/DLTK_IXI_Dataset/1mm/')
##adlsFileSystemClient.mkdir('/clusters/DLTK_IXI_Dataset/2mm/')
#
#Below tar file extraction take very long time, so use Databricks job scheduling if need to repeat

# COMMAND ----------

# DONT RUN THis NEXT TIME
# Note: This only need to extract one time!! no need to re-extract the tar files again.  
fnames = {}
fnames['T1'] = '/clusters/DLTK_IXI_Dataset/IXI-T1.tar'
fnames['T2'] = '/clusters/DLTK_IXI_Dataset/IXI-T2.tar'
fnames['MRA'] = '/clusters/DLTK_IXI_Dataset/IXI-MRA.tar'
fnames['PD'] = '/clusters/DLTK_IXI_Dataset/IXI-PD.tar'
##fnames['demographic'] = '/clusters/DLTK_IXI_Dataset/IXI.xls'

## Extract all the HH images to t1 or t2 directory.......
if EXTRACT_IMAGES:
    # Extract the HH subset of IXI
    for key, fname in fnames.items():
        if (fname.endswith('.tar')):
            print('Extracting IXI HH data from {}.'.format(fnames[key]))
            output_dir = os.path.join('/clusters/DLTK_IXI_Dataset', key)
            ## Create a output directory
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print("outputdir: ", output_dir)
            with adlsFileSystemClient.open(fnames[key], 'rb') as f:
                t = tarfile.open(name=key, fileobj=f, mode='r', debug=2)
                for member in t.getmembers():
                  if '-HH-' in member.name:
                     t.extract(member, output_dir)
                     ## Extract and store into the Datalakes Store new folder
                     multithread.ADLUploader(adlsFileSystemClient, lpath=output_dir, rpath=output_dir, nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)


# COMMAND ----------

if PROCESS_OTHER:
    # Process the demographic xls data and save to csv
    with adlsFileSystemClient.open('/clusters/DLTK_IXI_Dataset/IXI.xls', 'rb') as f:
      xls= pd.ExcelFile(f)
    print(xls.sheet_names)

    df = xls.parse('Table')
    for index, row in df.iterrows():
        IXI_id = 'IXI{:03d}'.format(row['IXI_ID'])
        df.loc[index, 'IXI_ID'] = IXI_id

        t1_exists = len(glob.glob('/clusters/DLTK_IXI_Dataset/T1/{}*.nii.gz'.format(IXI_id)))
        t2_exists = len(glob.glob('/clusters/DLTK_IXI_Dataset/T2/{}*.nii.gz'.format(IXI_id)))
        pd_exists = len(glob.glob('/clusters/DLTK_IXI_Dataset/PD/{}*.nii.gz'.format(IXI_id)))
        mra_exists = len(glob.glob('/clusters/DLTK_IXI_Dataset/MRA/{}*.nii.gz'.format(IXI_id)))

        # Check if each entry is complete and drop if not
        # if not t1_exists and not t2_exists and not pd_exists and not mra
        # exists:
        if not (t1_exists and t2_exists and pd_exists and mra_exists):
            print(IXI_id, t1_exists, t2_exists, pd_exists, mra_exists)
            df.drop(index, inplace=True)

    # Write to csv file (local file path)
    df.to_csv('/clusters/DLTK_IXI_Dataset/demographic_HH.csv', index=False)
    # upload the csv to the datalakestore
    multithread.ADLUploader(adlsFileSystemClient, lpath='/clusters/DLTK_IXI_Dataset/demographic_HH.csv', rpath='/clusters/DLTK_IXI_Dataset/demographic_HH.csv', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)


# COMMAND ----------

def resample_image(itk_image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0]*(original_spacing[0]/out_spacing[0]))),
                int(np.round(original_size[1]*(original_spacing[1]/out_spacing[1]))),
                int(np.round(original_size[2]*(original_spacing[2]/out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


def reslice_image(itk_image, itk_ref, is_label=False):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(itk_ref)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

# COMMAND ----------

if RESAMPLE_IMAGES:
    # Resample the IXI HH T2 images to 1mm isotropic and reslice all
    # others to it
    #df = pd.read_csv('/clusters/DLTK_IXI_Dataset/demographic_HH.csv', dtype=object, keep_default_na=False, na_values=[]).as_matrix()
    with adlsFileSystemClient.open('/clusters/DLTK_IXI_Dataset/demographic_HH.csv', 'rb') as f:
      df = pd.read_csv(f, dtype=object, keep_default_na=False, na_values=[]).as_matrix()

    for i in df:
        IXI_id = i[0]
        print('Resampling {}'.format(IXI_id))

        t1_fn = glob.glob('/clusters/DLTK_IXI_Dataset/T1/{}*.nii.gz'.format(IXI_id))[0]
        t2_fn = glob.glob('/clusters/DLTK_IXI_Dataset/T2/{}*.nii.gz'.format(IXI_id))[0]
        pd_fn = glob.glob('/clusters/DLTK_IXI_Dataset/PD/{}*.nii.gz'.format(IXI_id))[0]
        mra_fn = glob.glob('/clusters/DLTK_IXI_Dataset/MRA/{}*.nii.gz'.format(IXI_id))[0]

        t1 = sitk.ReadImage(t1_fn)
        t2 = sitk.ReadImage(t2_fn)
        pd = sitk.ReadImage(pd_fn)
        mra = sitk.ReadImage(mra_fn)

        # Resample to 1mm isotropic resolution
        t2_1mm = resample_image(t2)
        t1_1mm = reslice_image(t1, t2_1mm)
        pd_1mm = reslice_image(pd, t2_1mm)
        mra_1mm = reslice_image(mra, t2_1mm)

        output_dir = os.path.join('/clusters/DLTK_IXI_Dataset/1mm/', IXI_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('T1: {} {}'.format(t1_1mm.GetSize(), t1_1mm.GetSpacing()))
        print('T2: {} {}'.format(t2_1mm.GetSize(), t2_1mm.GetSpacing()))
        print('PD: {} {}'.format(pd_1mm.GetSize(), pd_1mm.GetSpacing()))
        print('MRA: {} {}'.format(mra_1mm.GetSize(), mra_1mm.GetSpacing()))

        sitk.WriteImage(t1_1mm, os.path.join(output_dir, 'T1_1mm.nii.gz'))
        sitk.WriteImage(t2_1mm, os.path.join(output_dir, 'T2_1mm.nii.gz'))
        sitk.WriteImage(pd_1mm, os.path.join(output_dir, 'PD_1mm.nii.gz'))
        sitk.WriteImage(mra_1mm, os.path.join(output_dir, 'MRA_1mm.nii.gz'))

        # Resample to 2mm isotropic resolution
        t2_2mm = resample_image(t2, out_spacing=[2.0, 2.0, 2.0])
        t1_2mm = reslice_image(t1, t2_2mm)
        pd_2mm = reslice_image(pd, t2_2mm)
        mra_2mm = reslice_image(mra, t2_2mm)

        output_dir = os.path.join('/clusters/DLTK_IXI_Dataset/2mm/', IXI_id)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('T1: {} {}'.format(t2_2mm.GetSize(), t1_2mm.GetSpacing()))
        print('T2: {} {}'.format(t2_2mm.GetSize(), t2_2mm.GetSpacing()))
        print('PD: {} {}'.format(pd_2mm.GetSize(), pd_2mm.GetSpacing()))
        print('MRA: {} {}'.format(mra_2mm.GetSize(), mra_2mm.GetSpacing()))

        sitk.WriteImage(t1_2mm, os.path.join(output_dir, 'T1_2mm.nii.gz'))
        sitk.WriteImage(t2_2mm, os.path.join(output_dir, 'T2_2mm.nii.gz'))
        sitk.WriteImage(pd_2mm, os.path.join(output_dir, 'PD_2mm.nii.gz'))
        sitk.WriteImage(mra_2mm, os.path.join(output_dir, 'MRA_2mm.nii.gz'))


# COMMAND ----------

os.listdir("/clusters/DLTK_IXI_Dataset/1mm")

# COMMAND ----------

multithread.ADLUploader(adlsFileSystemClient, lpath='/clusters/DLTK_IXI_Dataset/1mm', rpath='/clusters/DLTK_IXI_Dataset/1mm', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)

# COMMAND ----------

multithread.ADLUploader(adlsFileSystemClient, lpath='/clusters/DLTK_IXI_Dataset/2mm', rpath='/clusters/DLTK_IXI_Dataset/2mm', nthreads=64, overwrite=True, buffersize=4194304, blocksize=4194304)