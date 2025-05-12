
from oparea import OpArea


#last_known = (-66.693249, 41.630238)
#last_known2 = (-66.896444, 41.356677)
#hc144_csp = (-67.329518, 41.395129)
#helo = 'MH-60-6028'
#fixed_wing = 'HC-144-2302'

# last_known = (-79.695803, 26.518261)
# hc144_csp = (-79.990021,  26.063275)

last_known = (-80.026280,  26.219869)
helo = 'MH-65-6579'
helo_csp = (-80.079026, 26.139537)
#fixed_wing = 'HC-144-2316'

length = 7.5
width = 4
major_axis = 3

oparea = OpArea.from_datum(last_known, length, width, major_axis)

oparea.generate_sector_search(last_known, 1.25, 90, f'{helo}-01', n_patterns=2)
oparea.generate_parallel_track_search(helo_csp, 90, 0.5, f'{helo}-02')

oparea.to_kml(
    r'D:\Documents\code\projects\search_patterns\etc\oparea.kml',
    'TestOpArea',
    fill='orange',
    doc_name = 'CASE-20250504-002',
)
