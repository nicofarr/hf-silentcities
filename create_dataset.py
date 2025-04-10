import pandas as pd
import argparse
import os
from librosa import load
from datasets import Dataset, Audio
from tqdm import tqdm 

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description='Create a dataset for a given site')
parser.add_argument('--folder', type=str, help='Folder containing the base dataset')
parser.add_argument('--site', type=str, help='Site name (three digits)')
parser.add_argument('--metadata', type=str, help='Folder containing the metadata csv files')
parser.add_argument('--sitefile', type=str, help='site.csv file with the metadata of all sites')
args = parser.parse_args()

allsitedata = pd.read_csv(args.sitefile)
cursite = allsitedata.loc[allsitedata['partID']==f"partID{args.site}"].to_dict()


def transform(row):
    audio, sr = load(os.path.join(args.folder,args.site,row['name']),sr=None)
    row['array'] = {audio}
    return row

def add_sitedata(row):
    ## add cursite to row
    for key in cursite.keys():
        row[key] = cursite[key]
    return row
#
## Open the metadata file and directly convert it into a HuggingFace Dataset
ds = Dataset.from_pandas(pd.read_csv(os.path.join(args.metadata,f"partID{args.site}.csv.gz")))

### List all the tar.xz archives in the base dataset folder
archives = [f for f in os.listdir(os.path.join(args.folder)) if f.endswith(".tar.gz") and f.startswith(f"partID{args.site}")]
print(f"Archives found: {archives}")

### for each archive, extract it in the base dataset folder
for archive in archives:
    print(f"Uncompressing archive name {os.path.join(args.folder,archive)}")
    os.system(f"tar -xvf {os.path.join(args.folder,archive)} -C {os.path.join(args.folder,args.site)}")

    ## List all flac files corresponding to this archive
    allflacfiles = [f for f in os.listdir(os.path.join(args.folder)) if f.endswith(".flac")]

    ## split the main dataset by keeping only the rows that include allflacfiles as the name column
    print(f"Filtering the main dataset")
    ds_sub = ds.filter(lambda x: x['name'] in allflacfiles)

    fullpaths = [os.path.join(args.folder,f) for f in ds_sub['name']]

    print(f"Creating audio dataset and merging with columns")
    audio_dataset = Dataset.from_dict({"audio": fullpaths}).cast_column("audio", Audio())

    ds_sub = ds_sub.add_column("audio", audio_dataset["audio"])
    #print(f"Saving parquet file {archive}.parquet")
    #ds_sub.to_parquet(os.path.join(f"{archive}.parquet")) ### to check : parquet writer options pour la compression; eg snappy? 
    
    
    print("Pushing to hub...")
    ds_sub.push_to_hub(repo_id='nicofarr/silentcities-test',config_name=archive[:13])
    print(ds_sub)
    print(f"Removing flac files from disk")
    for flacfile in allflacfiles:
        os.remove(os.path.join(args.folder,flacfile))

    print(ds_sub.cleanup_cache_files())