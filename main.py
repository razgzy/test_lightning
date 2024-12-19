from lightning.pytorch.cli import LightningCLI
import torch
torch.set_printoptions(precision=4)

def cli_main():
    cli = LightningCLI(parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == '__main__':
    cli_main()
