# Augmented-Geometric-Distillation
Official code for Augmented Geometric Distillation

Datasets Structure
```
./data
- market
  - bounding_box_test
  - bounding_box_train
  - query
 
- msmt17
  - bounding_box_test
  - bounding_box_train
  - query

...
```

To train the baisc model on task $ T_1 $ (MSMT17), then run:
```
python manin.py -g 0 --dataset msmt17 --logs-dir ./logs/msmt17
```

To generation dreaming data via DeepInversion [1], then run:
```
python inversion.py -g 0 --generation-dir ./data/generations_r50_msmt17 --shots 40 --iters 640 --teacher ./logs/msmt17
```

To train the incremental model on task $ T_2 $ (Market) with Geometric Distillation loss, then run:
```
python main_incremental.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_GD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --algo-config ./configs/res-triangle.yaml
```

To train the incremental model on task $ T_2 $ (Market) with simple Geometric Distillation loss (detailed in Supp. and usually report better performance), then run:
```
python main_incremental.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_simGD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --algo-config ./configs/sim-res-triangle.yaml
```

To train the incremental model on task $ T_2 $ (Market) with Augmented Distillation, then run:
```
python main_incrementalX.py --dataset market --previous ./logs/msmt17 --logs-dir ./logs/msmt17-market_XsimGD --inversion-dir ./data/generations_r50_msmt17 -g 0 --evaluate 80 --seed 1 --peers 2 --algo-config ./configs/inverXion.yaml
```

If you want to reproduce other results in our work, just modify ` algo-config `.

Best wishes 🌈

[1] Yin, Hongxu, et al. "Dreaming to distill: Data-free knowledge transfer via deepinversion." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

