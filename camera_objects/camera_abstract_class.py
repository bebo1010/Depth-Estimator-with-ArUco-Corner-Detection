from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

class Two_Cameras_System(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_grayscale_images(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        return
    
    @abstractmethod
    def get_depth_image(self) -> Tuple[bool, np.ndarray]:
        return
    
    @abstractmethod
    def get_width(self) -> int:
        return
    
    @abstractmethod
    def get_height(self) -> int:
        return