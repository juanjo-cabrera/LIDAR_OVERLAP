from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from gmplot import GoogleMapPlotter
import pandas as pd
import numpy as np

class CustomGoogleMapPlotter(GoogleMapPlotter):
    def __init__(self, center_lat, center_lng, zoom, apikey='',
                 map_type='satellite'):
        if apikey == '':
            try:
                with open('apikey.txt', 'r') as apifile:
                    apikey = apifile.readline()
            except FileNotFoundError:
                pass
        super().__init__(center_lat, center_lng, zoom, apikey)

        self.map_type = map_type
        assert(self.map_type in ['roadmap', 'satellite', 'hybrid', 'terrain'])

    def write_map(self,  f):
        f.write('\t\tvar centerlatlng = new google.maps.LatLng(%f, %f);\n' %
                (self.center[0], self.center[1]))
        f.write('\t\tvar myOptions = {\n')
        f.write('\t\t\tzoom: %d,\n' % (self.zoom))
        f.write('\t\t\tcenter: centerlatlng,\n')

        # Change this line to allow different map types
        f.write('\t\t\tmapTypeId: \'{}\'\n'.format(self.map_type))

        f.write('\t\t};\n')
        f.write(
            '\t\tvar map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);\n')
        f.write('\n')

    def pos_scatter(self, lats, lngs):
        for lat, lon in zip(lats, lngs):
            self.scatter(lats=[lat], lngs=[lon], c='orange', size=0.3, marker=False,
                         s=None)

    def overlap_scatter(self, lats, lons, scan_idx, overlaps):
        def rgb2hex(rgb):
            """ Convert RGBA or RGB to #RRGGBB """
            rgb = list(rgb[0:3])  # remove alpha if present
            rgb = [int(c * 255) for c in rgb]
            hexcolor = '#%02x%02x%02x' % tuple(rgb)
            return hexcolor

        norm = Normalize(vmin=0, vmax=1, clip=True)
        mapper = ScalarMappable(norm=norm)
        mapper.set_array(overlaps)
        colors = [rgb2hex(mapper.to_rgba(value)) for value in overlaps]

        # sort according to overlap
        indices = np.argsort(overlaps)
        for i in indices:
            self.scatter(lats=[lats[i]], lngs=[lons[i]], c=colors[i], size=0.35, marker=False,
                         s=None)
        self.scatter(lats=[lats[scan_idx]], lngs=[lons[scan_idx]], c='red', size=0.17, marker='X',
                     s=None)

    def plot_trajectories(self, lats, lons, directory='/home/arvc/Escritorio/develop/Rosbags_Juanjo/Exterior_innova/map.html'):
        self.pos_scatter(lats, lons)
        self.draw(directory)

    def plot_overlap(self, lats, lons, scan_idx, overlaps, directory='/home/arvc/Escritorio/develop/Rosbags_Juanjo/Exterior_innova/map.html'):
        self.overlap_scatter(lats, lons, scan_idx, overlaps)
        self.draw(directory)


if __name__ == "__main__":
    initial_zoom = 20
    df = pd.read_csv('/home/arvc/Escritorio/develop/Rosbags_Juanjo/Exterior_innova/2022-12-07-15-22-43/robot0/gps0/data.csv')
    lats = df['latitude'].values.tolist()
    print(len(lats))
    lons = df['longitude'].values.tolist()
    print(len(lons))
    alt = df['altitude'].values.tolist()
    print(len(alt))
    gmap = CustomGoogleMapPlotter(lats[0], lons[0], initial_zoom,
                                  map_type='satellite')

    gmap.plot_trajectories(lats, lons, directory="/home/arvc/Escritorio/develop/Rosbags_Juanjo/Exterior_innova/2022-12-07-15-22-43/mymap.html")

