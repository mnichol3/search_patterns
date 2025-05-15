
from oparea import OpArea


#last_known = (-66.693249, 41.630238)
#last_known2 = (-66.896444, 41.356677)
#hc144_csp = (-67.329518, 41.395129)
#helo = 'MH-60-6028'
#fixed_wing = 'HC-144-2302'

# last_known = (-79.695803, 26.518261)
# hc144_csp = (-79.990021,  26.063275)

last_known = (-68.125874,  41.788893)
helo = 'MH-60-6031'
helo_csp = last_known
#fixed_wing = 'HC-144-2316'

length = 10
width = 5
major_axis = 15

oparea = OpArea.from_datum(last_known, length, width, major_axis)

#import pdb; pdb.set_trace()
oparea.generate_sector_search(last_known, 1.25, 90, f'{helo}-01', n_patterns=1)
#oparea.generate_parallel_track_search(helo_csp, 90, 0.5, f'{helo}-02')
oparea.generate_expanding_square_search(last_known, 135, 1, f'{helo}-03')
oparea.to_kml(
    r'C:\Users\mnicholson\code\etc\misc\oparea.kml',
    'TestOpArea',
    fill='orange',
    doc_name = 'CASE-20250504-002',
)
