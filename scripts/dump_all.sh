mkdir ~/.qlib/src_data/$1 && unzip ~/Downloads/$1.zip -d ~/.qlib/src_data/$1
python scripts/dump_bin.py dump_all --data_path ~/.qlib/src_data/$1/features/ --qlib_dir ~/.qlib/qlib_data/$1 --include_fields open,high,low,close,volume,factor
