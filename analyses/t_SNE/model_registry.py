from train import model_dict as _base_model_dict
from train_with_repa import model_dict as _repa_model_dict
from train_with_MoS_repa import model_dict as _mos_repa_model_dict


model_dict = {**_base_model_dict, **_repa_model_dict, **_mos_repa_model_dict}
