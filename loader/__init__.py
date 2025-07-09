from torch.utils import data
from loader.deepovec import deepovec

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict)
    
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", False),
        num_workers=data_dict["num_workers"],
    )
    return loader

def get_dataset(data_dict):
    name = data_dict["dataset"]
    
    if name == 'deepovec':
        dataset = deepovec(**data_dict)
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")
    
    return dataset