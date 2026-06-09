import argparse
import torch
import datetime
import json
import yaml
import os

from dataset_Synthetic import get_dataloader


from main_model import CSDI_Synthetic
from utils import train, evaluate

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False"
)


parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')


parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument(
    "--validationindex", type=int, default=0, help="index of month used for validation (value:[0-7])"
)

parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--unconditional", action="store_true", default=False)

############################################
parser.add_argument(
    "--targetstrategy", type=str, default="random", choices=["mix", "random", "block"]
)
parser.add_argument("--missing_pattern", type=str, default="point")  # block|point
parser.add_argument("--use_latent_diffusion_imputation", default=False)
parser.add_argument("--testmissingratio", type=float, default=0.5)


args = parser.parse_args()
# print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["target_strategy"] = args.targetstrategy
config["model"]["test_missing_ratio"] = args.testmissingratio


# print(json.dumps(config, indent=4))


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = (
        "./save/Synthetic_validationindex" + str(args.validationindex) + "_" + current_time + "/"
)

# print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    config["train"]["batch_size"], device=args.device, missing_pattern=args.missing_pattern,
    target_strategy=args.targetstrategy,
    missing_ratio=config["model"]["test_missing_ratio"],
    use_latent_diffusion_imputation=args.use_latent_diffusion_imputation,
)

model = CSDI_Synthetic(config, args.device).to(args.device)
if __name__ == '__main__':
    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

    evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
    )
