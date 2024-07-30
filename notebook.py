# %% [markdown]
# # Proyek Akhir: Pengembangan dan Pengoperasian Sistem Machine Learning

# %% [markdown]
# - Nama: Nicolas Debrito
# - Email: nicolas.debrito66@gmail.com
# - Id Dicoding: reezzy

# %% [markdown]
# ## Import Library

# %% [markdown]
# Melakukan import library yang dibutuhkan

# %%
import os
import pandas as pd
from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from sklearn.utils import resample
from modules.components import init_components

# %% [markdown]
# # Preparing the Dataset

# %% [markdown]
# Membuat dataframe dari csv

# %%
df = pd.read_csv('data/train.csv')
df

# %% [markdown]
# Melakukan encoding untuk kolom Offensive

# %%
df['Offensive'] = df['Offensive'].apply(lambda x: 0 if x == 'No' else 1)
df.head()

# %% [markdown]
# Menghitung jumlah data

# %%
df['Offensive'].value_counts()

# %% [markdown]
# Memisahkan data berdasarkan kolom offensive

# %%
df_minor = df[df['Offensive'] == 0]
df_mayor = df[df['Offensive'] == 1]

# %% [markdown]
# Melakukan upsampling pada data

# %%
df_upsampling = resample(df_minor, n_samples=len(df_mayor), random_state=42)
df = pd.concat([df_mayor, df_upsampling]).reset_index(drop=True)
df['Offensive'].value_counts()

# %% [markdown]
# Menyimpan dalam bentuk csv

# %%
df.to_csv('fix_data/Fix_Data.csv', index=False)

# %% [markdown]
# ## Run Pipeline

# %% [markdown]
# Membuat pipline yang akan menjalankan komponen pipeline menggunakan apache beam

# %%
PIPELINE_NAME = "hate-speech-pipeline"

DATA_ROOT = "fix_data"
TRANSFORM_MODULE_FILE = "modules/hate_speech_transform.py"
TRAINER_MODULE_FILE = "modules/hate_speech_trainer.py"

OUTPUT_BASE = "reezzy-pipeline"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        "----direct_num_workers=0" 
    ]
    
    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args
    )

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    
    components = init_components(
        DATA_ROOT,
        training_module=TRAINER_MODULE_FILE,
        transform_module=TRANSFORM_MODULE_FILE,
        serving_model_dir=serving_model_dir,
    )
    
    pipe = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipe)


