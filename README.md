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
