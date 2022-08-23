# PGN(Patch Generation Network)

## Train

```
python examples/train.py --config-file /path/to/config_file.yml 
```

If you want to output `DEBUG` level logs, you can modify `INFO.LOG_LEVEL` option in the config file or use:
```
python examples/train.py --config-file /path/to/config_file.yml INFO.LOG_LEVEL DEBUG
```
Logger with `DEBUG` level will output most of time consuming information.

## Eval

Simply get patches for a image:
```shell script
python examples/eval_single.py --config-file /path/to/config_file.yml --image_path /path/to/image.jpg
```

If you want to show recall evaluation result, just add `--load_panda_annotation` flag.
```shell script
python examples/eval_single.py --config-file /path/to/config_file.yml --image_path /path/to/image.jpg --load_panda_annotation
```
The annotation info for PANDA test set will be loaded according to the config in `config_file`, so be sure the `image_path` refers to an image in **PANDA test set**.

## Visualize
If you want to visualize the patches in the image, just add `--display` flag.
```shell script
python examples/eval_single.py --config-file /path/to/config_file.yml --image_path /path/to/image.jpg --display
```

If the operation system does not support the display mode, you can user the `--save_path` parameter to save result into a file:
```shell script
python examples/eval_single.py --config-file /path/to/config_file.yml --image_path /path/to/image.jpg --save_path result.jpg
```
 
