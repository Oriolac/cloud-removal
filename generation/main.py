import argparse
import cnn
import logging

import cnn.main
import ae.main
import dif.main
import gan.main
import vae.main

MAINS = {
    "cnn": cnn.main.main,
    "ae": ae.main.main,
    "gan": gan.main.main,
    "vae": vae.main.main,
    "dif": dif.main.main
}


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("config", type=str)
    kwargs = parser.parse_args().__dict__
    model_main = kwargs.pop("model")
    MAINS[model_main](**kwargs)


if __name__ == '__main__':
    main()
