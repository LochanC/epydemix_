from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from .simulation_output import Trajectory

@dataclass
class SimulationResults:
    """
    Class to store and manage multiple simulation results.
    
    Attributes:
        trajectories (List[Trajectory]): List of simulation trajectories
        parameters (Dict[str, Any]): Dictionary of parameters used in the simulations
    """
    trajectories: List[Trajectory]
    parameters: Dict[str, Any]

    @property
    def Nsim(self) -> int:
        """Number of simulations."""
        return len(self.trajectories)
    
    @property
    def dates(self) -> List[pd.Timestamp]:
        """Simulation dates."""
        return self.trajectories[0].dates if self.trajectories else []
    
    @property
    def compartment_idx(self) -> Dict[str, int]:
        """Compartment indices."""
        return self.trajectories[0].compartment_idx if self.trajectories else {}

    def get_stacked_trajectories(self) -> Dict[str, np.ndarray]:
        """
        Get trajectories stacked into arrays of shape (Nsim, timesteps, demographics).
        """
        if not self.trajectories:
            return {}
        
        return {
            comp_name: np.stack([t.data[comp_name] for t in self.trajectories], axis=0)
            for comp_name in self.trajectories[0].data.keys()
        }

    def get_quantiles(self, quantiles: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Compute quantiles across all trajectories.
        """
        if quantiles is None:
            quantiles = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]

        stacked = self.get_stacked_trajectories()
        
        # Create dates and quantiles first (these will be the same for all compartments)
        dates = []
        quantile_values = []
        for q in quantiles:
            dates.extend(self.dates)
            quantile_values.extend([q] * len(self.dates))
        
        # Initialize data dictionary with dates and quantiles
        data = {
            "date": dates,
            "quantile": quantile_values
        }
        
        # Add compartment data
        for comp_name, comp_data in stacked.items():
            comp_quantiles = []
            for q in quantiles:
                quant_values = np.quantile(comp_data, q, axis=0)
                comp_quantiles.extend(quant_values)
            data[comp_name] = comp_quantiles

        return pd.DataFrame(data)

    def resample(self, freq: str, method: str = 'last', fill_method: str = 'ffill') -> 'SimulationResults':
        """Resample all trajectories to new frequency."""
        return SimulationResults(
            trajectories=[t.resample(freq, method, fill_method) for t in self.trajectories],
            parameters=self.parameters
        )
