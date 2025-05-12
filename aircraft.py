from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from convert import NMI_2_M


@dataclass(kw_only=True, repr=False)
class Aircraft:
    name: str
    endurance: int
    range: int
    radius: int
    radar: str
    vsweep_widths: pd.DataFrame
    vsweep_corrections: pd.DataFrame

    def get_vis_sweep_width(
        self,
        target: str,
        alt: int,
        vis: int,
        gspeed: int | None = None,
        wind: int | None = None,
        seas: int | None = None,
    ) -> float:
        """Calculate the corrected visual sweep width, in meters.

        Parameters
        ----------
        TODO

        Returns
        -------
        float
        """
        wind = wind if wind is not None else 0
        seas = seas if seas is not None else 0

        target, size = target.split('_')

        vis = min(
            [1, 3, 5, 10, 15, 20, 30],
            key=lambda x: abs(x - vis))

        ref_vis = f'visibility-{vis}'

        ref_alt = min(
            self.vsweep_widths['altitude'].unique(),
            key=lambda x: abs(x - alt))

        width = self.vsweep_widths.loc[
            (self.vsweep_widths['object'] == target) \
            & (self.vsweep_widths['size'] == int(size)) \
            & (self.vsweep_widths['altitude'] == ref_alt)
        ][ref_vis]

        # TODO: speed & wx correction

        return width * NMI_2_M

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.name}>'


@dataclass(kw_only=True, repr=False)
class FixedWing(Aircraft):
    category: ClassVar[str] = 'fixed-wing'


@dataclass(kw_only=True, repr=False)
class Helicopter(Aircraft):
    category: ClassVar[str] = 'helicopter'
