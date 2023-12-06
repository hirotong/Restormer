import os
import sys
import torch
import yaml

from basicsr.models.archs import define_network
from basicsr.models import create_model
from basicsr.utils.options import ordered_yaml

assert len(sys.argv) == 4, "Args are wrong"

input_path = sys.argv[1]
output_path = sys.argv[2]
config_path = sys.argv[3]


def get_node_name(name, parent_name):
    if len(name) < len(parent_name):
        return False, ""
    p = name[: len(parent_name)]
    if p != parent_name:
        return False, ""
    return True, name[len(parent_name) :]


with open(config_path, "r") as f:
    opt = yaml.load(f, Loader=ordered_yaml()[0])


network_g = define_network(opt['network_g'])

pretrained_weights = torch.load(input_path)
if "state_dict" in pretrained_weights:
    pretrained_weights = pretrained_weights["state_dict"]
if "params" in pretrained_weights:
    pretrained_weights = pretrained_weights['params']

scratch_dict = network_g.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, "control.")
    if is_control:
        copy_k = name
    else:
        _, name = get_node_name(k, "restormer.")
        copy_k = name
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        print(f"These weights are newly added: {k}")

network_g.load_state_dict(target_dict, strict=True)

save_dict = {"params": network_g.state_dict()}

torch.save(save_dict, output_path)
print("Done")
        
