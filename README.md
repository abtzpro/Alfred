# Alfred
Cybersecurity Oriented Quantum AI | QML | QNN

## Training Dataset Notes

You can replace 'dest-dir' with the path to the folder where you want to download the training dataset. If the dataset is inside a folder in the AWS bucket, you need to replace '' in folder_in_bucket with the name of the folder. The script will download each file in the bucket to your specified folder.

The script was tested and built around using CSE-CIC-IDS2018 dataset from the Canadian Institute for Cybersecurity, which includes data on more modern attack types. This dataset and install instructions can be found here: https://www.unb.ca/cic/datasets/ids-2018.html

The script has been modified to work with the following CSE-CIC-IDS2018 Dataset install methodology 
```
Install the AWS CLI, available on Mac, Windows and Linux
Run: aws s3 sync --no-sign-request --region <your-region> "s3://cse-cic-ids2018/" dest-dir (Where your-region is your region from the AWS regions list and dest-dir is the name of the desired destination folder in your machine)
```

The above dataset installation methodology has been modified and adapted in such a way that a user only has to edit two placeholder values prior to the script's first run in order to automatically fetch, download, preprocess and utilize the aforementioned cybersecurity oriented training dataset. 
