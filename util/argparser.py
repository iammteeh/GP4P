# define a simple interface for parsing command line arguments
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

class ArgParseInterface(ABC):
    def __init__(self, parser: ArgumentParser) -> None:
        self.parser = parser
        self.add_arguments()

    def parse_args(self) -> Any:
        return self.parser.parse_args()
    
    @abstractmethod
    def add_arguments(self) -> None:
        raise NotImplementedError