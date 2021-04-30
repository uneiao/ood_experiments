# Temporal repo for ood experiments

## Training

First open `src` directory.
```
cd src && python3 main.py...
```
### Initial run
```
python3 main.py --mode train --config_file configs/default.yaml resume False
```
### Resume run 
```
python3 main.py --mode train --config_file configs/default.yaml
```

## Logs

Logs will be generated in `./output/logs`. One can view the visualization by
```
tensorboard --logdir output/logs/
```
