### Usage
Generate up to 500 C files from the parquet export:

Indicating which scores are counted as "vulnerable"
```
python export_c_files.py export_25-06-05.parquet --positive-scores NOT_HELPFUL UNREACHED SATURATED -n 500
```

```shell
bash run_aijon_joern_parse.sh NOT_HELPFUL_UNREACHED_SATURATED_500
```