from typing import *

from functools import partial
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import numpy as np

from .abstract import Elasticity, Plasticity
import src.constitutive_models.physical_constitutive_models as physical

class LLM_Elasticity(Elasticity):
    def __init__(
        self,
        physicals: List[str],
        obj_idx: List[int], 
        device: torch.device='cuda',
        
    ) -> None:
        super().__init__()
        self.physical_names = physicals
        self.physicals = nn.ModuleList([getattr(physical, p)().to(device) for p in physicals])
        self.category_dim = len(self.physicals)
        self.device = device
        self.obj_idx = obj_idx

    def forward(self, F: Tensor, labels, log_E) -> Tensor:

        # 先创建一个stress tensor
        total_stress = torch.zeros((F.shape[0], 3, 3), device=self.device)

        # 每一次是m* 3 *3, 保证所有的执行完都是n*3*3
        for i, idx in enumerate(self.obj_idx):
            # 有几个物体，就有几个势能模型，在参数配置里改
            sub_model = self.physicals[i]

            # 每个物体都有一个label
            # 挑选出属于该物体的点
            sub_mask = (labels == idx)
            sub_points = F[sub_mask]

            # 得到sub应力
            sub_stress = sub_model(sub_points, log_E[i])
            total_stress[sub_mask] = sub_stress

        # 其余的点, 全部默认模型0
        # 计算其余点的应力
        res_mask = ~np.isin(labels, self.obj_idx)  
        res_mask_tensor = torch.from_numpy(res_mask).to(F.device)
        res_points = F[res_mask_tensor]
        res_stress = self.physicals[0](res_points, log_E[0])  # 使用默认模型
        total_stress[res_mask] = res_stress

        #  N * 3 * 3 
        return total_stress   
      
    def name(self) -> str:
        return f'LLM_Elasticity among {", ".join(self.physical_names)}'


class LLM_Plasticity(Plasticity):
    def __init__(
        self,
        physicals: List[str],
        obj_idx: List[int], 
        device: torch.device='cuda',
    ) -> None:
        super().__init__()
        self.physical_names = physicals
        self.physicals = nn.ModuleList([getattr(physical, p)().to(device) for p in physicals])
        self.category_dim = len(self.physicals)
        self.device = device
        self.obj_idx = obj_idx

    def forward(self, F: Tensor, labels, log_E) -> Tensor:
        sub_model = self.physicals[0]

        possible_stress = sub_model(F, log_E[0])
        
        stress = possible_stress
        return stress
    
    def name(self) -> str:
        return f'LLM_Plasticity among {", ".join(self.physical_names)}'

     
def hard_softmax(logits: Tensor, dim: int) -> Tensor:
    y_soft = logits.softmax(dim=dim)
    index = y_soft.argmax(dim=dim, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(dim=dim, index=index, value=1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret

class GumbelElasticity(Elasticity):
    def __init__(
        self,
        physicals: List[str],
        device: torch.device='cuda',
    ) -> None:
        super().__init__()
        self.physical_names = physicals
        self.physicals = nn.ModuleList([getattr(physical, p)().to(device) for p in physicals])
        self.category_dim = len(self.physicals)
        
    def forward(self, F: Tensor, elasticity_category: Tensor) -> Tensor:
        assert elasticity_category.shape[1] == self.category_dim
        # max_category = torch.functional.F.gumbel_softmax(elasticity_category, tau=1.0, dim=1, hard=True) # num_particles * category_dim
        max_category = hard_softmax(elasticity_category, dim=1)
        max_category = max_category.unsqueeze(dim=2).unsqueeze(dim=3) # num_particles * category_dim * 1 * 1
        possible_stress = [p(F) for p in self.physicals] # category_dim * num_particles * 3 * 3
        possible_stress = torch.stack(possible_stress, dim=1) # num_particles * category_dim * 3 * 3
        stress = possible_stress * max_category # num_particles * category_dim * 3 * 3
        stress = stress.sum(dim=1) # num_particles * 3 * 3
        return stress
    
    def name(self) -> str:
        return f'[Elasticity] Neural GumbelElasticity among {", ".join(self.physical_names)}'

class GumbelPlasticity(Plasticity):
    def __init__(
        self,
        physicals: List[str],
        device: torch.device='cuda',
    ) -> None:
        super().__init__()
        self.physical_names = physicals
        self.physicals = nn.ModuleList([getattr(physical, p)().to(device) for p in physicals])
        self.category_dim = len(self.physicals)
        
    def forward(self, F: Tensor, plasticity_category: Tensor) -> Tensor:
        assert plasticity_category.shape[1] == self.category_dim
        # max_category = torch.functional.F.gumbel_softmax(plasticity_category, tau=5.0, dim=1, hard=True) # num_particles * category_dim
        max_category = hard_softmax(plasticity_category, dim=1)
        max_category = max_category.unsqueeze(dim=2).unsqueeze(dim=3) # num_particles * category_dim * 1 * 1
        possible_stress = [p(F) for p in self.physicals] # category_dim * num_particles * 3 * 3
        possible_stress = torch.stack(possible_stress, dim=1) # num_particles * category_dim * 3 * 3
        stress = possible_stress * max_category # num_particles * category_dim * 3 * 3
        stress = stress.sum(dim=1) # num_particles * 3 * 3
        return stress   
      
    def name(self) -> str:
        return f'[Plasticity] Neural GumbelPlasticity among {", ".join(self.physical_names)}'