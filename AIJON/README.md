### Usage

Indicating which scores are counted as "vulnerable" and the number of examples to process:
```shell
python export_c_files.py export_25-06-05.parquet --positive-scores NOT_HELPFUL UNREACHED SATURATED -n 500
```

```shell
bash run_aijon_joern_parse.sh NOT_HELPFUL_UNREACHED_SATURATED_500
```

run extract_slices and change directory if needed

create word2vec with train_wv_sysevr.ipynb -> need to adapt pathes

```shell
python ../data_processing/create_ggnn_data.py \
  --project NOT_HELPFUL_UNREACHED_SATURATED_500 

```
this will create ggnn.json.shard1 