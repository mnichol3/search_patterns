This Python module constructs parallel-track and sector search patterns.

## Requirements
* Python >= 3.11
* NumPy
* Pandas
* PyPROJ
* Shapely
* [UTM](https://pypi.org/project/utm/)

## Usage
```python
from oparea import OpArea

last_known = (-80.026280,  26.219869)
helo_id = 'MH-65-6579'
csp = (-80.079026, 26.139537)

oparea = OpArea.from_datum(last_known, 7.5,  4, 30)
oparea.generate_parallel_track_search(csp, 90, 0.5, f'{helo_id}-02')

oparea.to_kml(
    r'search_patterns\etc\oparea.kml',
    'TestOpArea',
    doc_name='CASE-20250504-002',
)
```
![Resulting parallel-track search pattern](https://github.com/mnichol3/search_patterns/blob/master/etc/example-01.jpg)


## Requirments
TODO
