gcp_project_id = 'kaggle-playground-170215'

gcs_path = "gs://{}-vcm/recursion-cellular-image-classification/RGB224/".format(gcp_project_id)

train_filename = "automl_train.csv"

gcp_service_account_json = '/kaggle/input/gcloudserviceaccountkey/kaggle-playground-170215-4ece6a076f22.json'

gcp_compute_region = 'us-central1' #for now, AutoML is only available in this region

train_budget = 24

dataset_name = 'recursion_224px_wo_controls'

model_name = "{}_{}".format(dataset_name,train_budget) 

#google cloud SDK







#AutoML package


#authenticate the Google Cloud SDK





#uncomment if you don't already have this gcs bucket setup

#!gsutil mb -p $gcp_project_id -c regional -l $gcp_compute_region gs://$gcp_project_id-vcm/



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import zipfile

import os

from google.cloud import automl_v1beta1 as automl



df_train = pd.read_csv('../input/recursion-2019-load-resize-and-save-images/new_train.csv')



validation = ['HEPG2-07','HUVEC-16','RPE-07','U2OS-02']

test = ['HEPG2-03','HUVEC-04','RPE-05','U2OS-01']





df_train['split'] = 'TRAIN' 

#put experiments in the lists aboe in Validation and TEST

df_train.loc[df_train['experiment'].isin(validation),'split'] = 'VALIDATION' 

df_train.loc[df_train['experiment'].isin(test),'split'] = 'TEST'



#add gcs path

df_train['gcspath'] = gcs_path + 'train/' + df_train['filename']



#AutoML requires the label to be an int not a float

df_train['sirna'] = df_train['sirna'].astype(int)



df_train[['split','gcspath','sirna']].to_csv(train_filename,index=False,header=False)


with zipfile.ZipFile('../input/recursion-2019-load-resize-and-save-images/train.zip', 'r') as zip_ref:

    zip_ref.extractall('./train/')


#1. Setup AutoML Data Object

client = automl.AutoMlClient.from_service_account_json(gcp_service_account_json)

project_location = client.location_path(gcp_project_id, gcp_compute_region)



my_dataset = {

    "display_name": dataset_name,

    "image_classification_dataset_metadata": {"classification_type": "MULTICLASS"},

}



# Create a dataset with the dataset metadata in the region

dataset = client.create_dataset(project_location, my_dataset)

dataset_id = (dataset.name.split("/")[-1])
#2 Load data into the object

dataset_full_id = client.dataset_path(

    gcp_project_id, gcp_compute_region, dataset_id

)



input_uris = ('{}{}'.format(gcs_path ,train_filename)).split(",")

input_config = {"gcs_source": {"input_uris": input_uris}}



response = client.import_data(dataset_full_id, input_config)



print("Processing import...")

print("Data imported. {}".format(response.result()))

#3. Train the model

my_model = {

    "display_name": model_name,

    "dataset_id": dataset_id,

    "image_classification_model_metadata": {"train_budget": train_budget}

    if train_budget

    else {},

}



response = client.create_model(project_location, my_model)



print("Training operation name: {}".format(response.operation.name))

print("Training started...")