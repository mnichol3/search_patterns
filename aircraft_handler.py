import json
from pathlib import Path

import pandas as pd

from aircraft import FixedWing, Helicopter


class AircraftHandler:

    categories = [
        'helicopter',
        'fixed_wing',
    ]

    data_dir = Path(__file__).resolve().parent.joinpath('etc')

    def __init__(self):
        self.aircraft = self._read_aircraft()
        self.vis_sweep_widths = self._read_vis_sweep_widths()
        self.vis_sweep_corrs = self._read_vis_sweep_corrs()

    def get_aircraft(self, name: str) -> FixedWing | Helicopter:
        name = name.upper()
        data = self.aircraft.get(name).copy()

        if not data:
            raise ValueError(f'Unknown aircraft "{name}"')

        if data['category'] == 'helicopter':
            data.pop('category')
            return Helicopter(
                **data,
                vsweep_widths=self.vis_sweep_widths['helicopter'],
                vsweep_corrections=self.vis_sweep_corrs['helicopter'],
            )
        else:
            data.pop('category')
            return FixedWing(**data)

    def _read_vis_sweep_corrs(self) -> dict:
        """Read visual sweep width correction data CSV files."""
        files = [
            'sweep_width-speed_corr-helicopter.csv',
            'sweep_width-speed_corr-fixed_wing.csv',
        ]

        return {
            k: pd.read_csv(self.data_dir.joinpath(f))
            for k, f in zip(self.categories, files)
        }

    def _read_vis_sweep_widths(self) -> dict:
        """Read visual sweep width data CSV files."""
        files = [
            'sweep_width-visual-helicopter.csv',
            'sweep_width-visual-fixed_wing.csv',
        ]

        return {
            k: pd.read_csv(self.data_dir.joinpath(f))
            for k, f in zip(self.categories, files)
        }

    def _read_aircraft(self) -> dict:
        """Read aircraft metadata from the aircraft.json file."""
        with open(self.data_dir.joinpath('aircraft.json'), 'r') as f:
            return json.load(f)
