from typing import *

import src.constitutive_models.physical_constitutive_models as physical
from .neural_constitutive_models import *

def get_elasticity(elasticity_name: str, physicals: Optional[List[str]]=None, device: str='cuda', obj_idx: List=None):
    if elasticity_name == 'neural':
        return GumbelElasticity(
            physicals=physicals,
            device=device
        )
    elif elasticity_name == 'LLM':
        return LLM_Elasticity(
            physicals=physicals,
            device=device, 
            obj_idx=obj_idx
            
        )
    else:
        return PresetElasticity(elasticity_name, device=device)

def get_plasticity(plasticity_name: str, physicals: Optional[List[str]]=None, device: str='cuda', obj_idx: List=None):
    if plasticity_name == 'neural':
        return GumbelPlasticity(
            physicals=physicals,
            device=device
        )
    elif plasticity_name == 'LLM':
        return LLM_Plasticity(
            physicals=physicals,
            device=device, 
            obj_idx=obj_idx
        )
    else:
        return PresetPlasticity(plasticity_name, device=device)

