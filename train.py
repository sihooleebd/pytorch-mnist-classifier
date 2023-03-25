import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImageClassifier
from trainer import Trainer
from utils import load_mnist
from utils import split_data
from utils import get_hidden_sizes


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True)
    p.add_argument("--train_ratio", type=float, default=0.8)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=20)

    p.add_argument("--n_layers", type=int, default=5)
    p.add_argument("--verbose", type=int, default=1)

    config = p.parse_args()

    return config


def main(config):
    x, y = load_mnist(is_train=True, flatten=True)
    x, y = split_data(x, y, train_ratio=config.train_ratio)

    print("Train:", x[0].shape, y[0].shape)
    print("valid:", x[1].shape, y[1].shape)

    input_size = int(x[0].shape[-1])
    output_size = int(max(y[0])) + 1

    model = ImageClassifier(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=get_hidden_sizes(input_size, output_size, config.n_layers),
        use_batch_norm=True,
    )
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model, optimizer, crit)

    trainer.train(train_data=(x[0], y[0]), valid_data=(x[1], y[1]), config=config)

    torch.save(
        {
            "model": trainer.model.state_dict(),
            "opt": optimizer.state_dict(),
            "config": config,
        },
        config.model_fn,
    )


if __name__ == "__main__":
    config = define_argparser()
    main(config)
