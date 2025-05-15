
from statistics import mean

import matplotlib.pyplot as plt
from shapely import Polygon

from tmtransformer import TMTransformer
from _oparea import TMOpArea, UTMOpArea


def get_mean_lonlat(coords: list[tuple[float, float]]) -> tuple[float, float]:
    lons, lats = list(zip(*coords))

    return mean(lons), mean(lats)


# coords = [
#     (-60.643460065636205, 41.51467803651396),
#     (-61.97876193436379, 41.51467803651398),
#     (-61.968704591861304, 40.51486150081292),
#     (-60.65351740813868, 40.51486150081292),
# ]

datum = (-68.125874,  41.788893)

# Append first coordinate to end for closure
#coords.append(coords[0])

# transformer = TMTransformer.from_lonlat(*get_mean_lonlat(coords))
# tm_coords = list(zip(*transformer.fwd(coords)))

# oparea = Polygon(tm_coords)

#oparea = TMOpArea(coords, 0.0, transform=True)
oparea = TMOpArea.from_datum(datum, 30, 15, 15)
# oparea.generate_parallel_track_search(
#     (-61.968704591861304, 40.51486150081292),
#     90,
#     3,
#     'SCORE31',
#     first_course=0,
# )

oparea.generate_sector_search(datum, 90, 3, 'SCORE31')
oparea.generate_expanding_square_search(
    datum,
    90,
    3,
    'SCORE32',
)

oparea.to_kml(
    r'C:\Users\mnicholson\code\etc\misc\oparea.kml',
    'TestOpArea',
    fill='orange',
    doc_name = 'CASE-20250504-002',
)
