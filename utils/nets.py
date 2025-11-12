from pathlib import Path
import torch
from torch import load
from nets.resnet import resnet50
from nets.clip import ClipLinear

def load_weights(model, model_path):
    dat = load(model_path, map_location='cpu')
    if 'model' in dat:
        if ('module._conv_stem.weight' in dat['model']) or \
           ('module.fc.fc1.weight' in dat['model']) or \
           ('module.fc.weight' in dat['model']):
            model.load_state_dict(
                {key[7:]: dat['model'][key] for key in dat['model']})
        else:
            model.load_state_dict(dat['model'])
    elif 'model_state_dict' in dat:
        from torch import cat
        model.load_state_dict({_[7:]: dat['model_state_dict'][_] if _[7:]!='conv1.weight' else cat((dat['model_state_dict'][_],dat['model_state_dict'][_],dat['model_state_dict'][_]),1)/3.0   for _ in  dat['model_state_dict']})
    elif 'state_dict' in dat:
        model.load_state_dict(dat['state_dict'])
    elif 'net' in dat:
        model.load_state_dict(dat['net'])
    elif 'main.0.weight' in dat:
        model.load_state_dict(dat)
    elif '_fc.weight' in dat:
        model.load_state_dict(dat)
    elif 'conv1.weight' in dat:
        model.load_state_dict(dat)
    else:
        print(list(dat.keys()))
        assert False
    return model

def load_network(namenet, weights_dir):
    
    model_path = Path(weights_dir, f'{namenet}.pth')
    if namenet == 'GRAG_latent_r50':
        model = resnet50(num_classes=1, stride0=1)
        model = load_weights(model, model_path)
    elif namenet == 'CORV_latent_r50':
        model = resnet50(num_classes=1, stride0=1)
        model = load_weights(model, model_path)
    elif namenet == 'WANG_latent_r50':
        model = resnet50(num_classes=1, stride0=2)
        model = load_weights(model, model_path)
    elif namenet == 'OJHA_latent_clip':
        model = ClipLinear()
        model = load_weights(model, model_path)
    else:
        assert False
    return model