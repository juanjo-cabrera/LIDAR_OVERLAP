import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from gmplot import GoogleMapPlotter
import pandas as pd

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
            self.scatter(lats=[lat], lngs=[lon], c='orange', size=0.17, marker=False,
                         s=None)



    def plot_trajectories(self, lats, lons, directory='/home/arvc/Escritorio/develop/Rosbags_Juanjo/Exterior_innova/map.html'):
        # initial_zoom = 20
        # directory = "/home/arvc/Escritorio/develop/Rosbags_Juanjo/Exterior_innova/2022-12-07-15-22-43/mymap.html"
        # gmap = CustomGoogleMapPlotter(lats[0], lons[0], initial_zoom,
        #                               map_type='satellite')
        gmap.pos_scatter(lats, lons)
        gmap.draw(directory)







initial_zoom = 20
#df = pd.read_csv('/home/arvc/DATASETS/gps_RTK/robot0/gps0/data.csv')
df = pd.read_csv('/home/arvc/Escritorio/develop/Rosbags_Juanjo/Exterior_innova/2022-12-07-15-22-43/robot0/gps0/data.csv')
lats=df['latitude'].values.tolist()
print(len(lats))
lons=df['longitude'].values.tolist()
print(len(lons))
alt=df['altitude'].values.tolist()
print(len(alt))



gmap = CustomGoogleMapPlotter(lats[0], lons[0], initial_zoom,
                              map_type='satellite')

gmap.plot_trajectories(lats, lons, directory="/home/arvc/Escritorio/develop/Rosbags_Juanjo/Exterior_innova/2022-12-07-15-22-43/mymap.html")
# gmap.color_scatter(lats, lons)

#gmap.draw("/home/arvc/DATASETS/gps_RTK/mymap.html")
# gmap.draw("/home/arvc/Escritorio/develop/Rosbags_Juanjo/Exterior_innova/2022-12-07-15-22-43/mymap.html")