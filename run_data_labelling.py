from run_ekf_remote_overlap import compute_overlap
import csv
from config import PARAMETERS
from eurocreader.eurocreader_outdoors import EurocReader


directory = PARAMETERS.directory
euroc_read = EurocReader(directory=directory)
scan_times, _, _ = euroc_read.prepare_ekf_data(deltaxy=PARAMETERS.exp_deltaxy,
                                                deltath=PARAMETERS.exp_deltath,
                                                nmax_scans=PARAMETERS.exp_long)





with open('/home/arvc/Escritorio/develop/Rosbags_Juanjo/Exterior_innova/prueba1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Reference Scan", "Others Scans", "Overlap"])
    for scan_idx in range(0, len(scan_times)):
        overlaps, scan_times_reference, scan_times_others = compute_overlap(scan_idx)
        for i in range(0, len(overlaps)):
            writer.writerow([scan_times_reference[i], scan_times_others[i], overlaps[i]])


