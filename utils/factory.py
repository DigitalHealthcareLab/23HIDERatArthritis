
from models.Hi_DER import HiDER

def get_model(model_name, args):
    name = model_name.lower()
    if name == "hider":
        return HiDER(args)
    else:
        assert 0
