
import sys
sys.path.insert( 0, '../perception/' )

from PUtils import *

_mu, _sigma = 100, 15
_x = _mu + _sigma * np.random.randn( 100000 )

_hist1, _ = np.histogram( _x, bins = 100, range = ( 0, 200 ) )
_hist1 = normalizeHistogram( _hist1 )

# _hist2 = normalizeHistogram( hist2hist( _hist1, 50, 0, 200 ) )
# _hist3 = normalizeHistogram( hist2hist( _hist1, 25, 0, 200 ) )
_hist2 = hist2hist( _hist1, 50 )
_hist3 = hist2hist( _hist1, 25 )

plotHistogram( _hist1, 0, 200 )
plotHistogram( _hist2, 0, 200 )
plotHistogram( _hist3, 0, 200 )

plt.show()