import torch
from torch import is_tensor
from torch.nn.utils.rnn import pad_sequence

# custom collater

def first(it):
    return it[0]

def custom_collate(data, pad_id = -1):
    is_dict = isinstance(first(data), dict)

    if is_dict:
        keys = first(data).keys()
        data = [d.values() for d in data]

    output = []

    for datum in zip(*data):
        if is_tensor(first(datum)):
            if first(datum).dtype == torch.bool:
                datam = pad_sequence(datum, batch_first = True, padding_value = False)
            else:
                datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)
        else:
            datum = list(datum)

        output.append(datum)

    output = tuple(output)

    if is_dict:
        output = dict(zip(keys, output))

    return output


def custom_collate2(data, pad_id = -1, split_id = -2):
    keys = data[0]['hair'][0].keys()

    new_data = []
    for item in data:
        hair = item['hair']
        new_hair = { k: [] for k in keys }
        for i, d in enumerate(hair):
            for k in keys:
                new_hair[k].append(d[k])
                if i != len(hair) - 1:
                    if len(d[k].shape) == 1:
                        new_hair[k].append(torch.tensor([split_id], device=d[k].device))
                    else:
                        new_hair[k].append(torch.ones((1, d[k].shape[1]), device=d[k].device) * split_id )
        for k in keys:
            new_hair[k] = torch.cat(new_hair[k], dim=0)
        for k in item.keys():
            if k != 'hair':
                new_hair[k] = item[k]
        new_data.append(new_hair)

    return custom_collate(new_data, pad_id=pad_id)
