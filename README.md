# test_lightning
only for debug

# environment
lightning                2.4.0

lightning-utilities      0.11.8

pytorch-lightning        2.4.0

torch                    2.4.1+cu124

torchvision              0.19.1+cu124

# command
run an experiment

```
python main.py fit --config config.yaml
```
load ckpt to continue: uncomment the ckpt_path in config.yaml
```
ckpt_path: runs/test/version_0/checkpoints/epoch=11-epoch=11.0000.ckpt
```
see logs
```
tensorboard --logdir=runs
```