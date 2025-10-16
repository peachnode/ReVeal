### Usage

First we have to export from our parquet file examples. From each line a c function file is created with the annotation included.
The file name ending indicates whether the function is vulnerable or not.
We have to decide which scores should be counted as vulnerable by setting the --positives-scores flag. Additionally we
can set a maximum amount of examples for testing. A directory with the scores used and the number of examples will be created
containg the function in the raw_code folder:
```shell
python export_c_files.py export_25-06-05.parquet --positive-scores NOT_HELPFUL UNREACHED SATURATED -n 500
```
This will create joern parsed files for every function from data/<PROJECT NAME>/raw_code in data/<PROJECT NAME>/parsed.
For every function an edges.csv and nodes.csv is created.

```shell
bash run_aijon_joern_parse.sh NOT_HELPFUL_UNREACHED_SATURATED_500
```

run extract_slices.ipynb #check
```shell
PROJECT="NOT_HELPFUL_UNREACHED_SATURATED_500" \
jupyter nbconvert --to notebook --execute ../data_processing/extract_slices.ipynb 

```

create word2vec with train_wv_sysevr.ipynb

```shell
PROJECT="NOT_HELPFUL_UNREACHED_SATURATED_500" \
jupyter nbconvert --to notebook --execute ../data_processing/train_wv_sysevr.ipynb
```

create ggnn.json.shard1 #check

```shell
python ../data_processing/create_ggnn_data.py \ 
  --project NOT_HELPFUL_UNREACHED_SATURATED_500 

```

```shell
PROJECT="NOT_HELPFUL_UNREACHED_SATURATED_500" \
jupyter nbconvert --to notebook --execute ../data_processing/full_data_prep_script.ipynb 

```
this create full_graph.json (and line-ggnn.json)

```shell
python ../data_processing/split_full_graph.py \
  --input ./data/NOT_HELPFUL_UNREACHED_SATURATED_500/NOT_HELPFUL_UNREACHED_SATURATED_500-full_graph.json \
  --output-dir ./data/NOT_HELPFUL_UNREACHED_SATURATED_500/prepared_devign \
  --train-ratio 0.8 --valid-ratio 0.1 --test-ratio 0.1
```


Run Devign. This will create AIJON/models/NOT_HELPFUL_UNREACHED_SATURATED_500/GGNNSumModel-model.bin.
```shell
python ../Devign/main.py \
    --dataset NOT_HELPFUL_UNREACHED_SATURATED_500 \
    --input_dir ./data/NOT_HELPFUL_UNREACHED_SATURATED_500/prepared_devign \
    --model_type devign \
    --node_tag node_features \
    --graph_tag graph \
    --label_tag targets \
    --feature_size 169

```