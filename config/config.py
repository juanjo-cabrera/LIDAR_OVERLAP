"""
Main config file of debugging, experiments, ICP and training parameters.
"""
import yaml

class ICP_ParametersConfig():
    """
    Clase en la que se almacenan los parametros del registration
    """
    def __init__(self, yaml_file='config/icp_parameters.yaml'):

        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)
            self.voxel_size = config.get('down_sample').get('voxel_size')
            self.max_distance = config.get('filter_by_distance').get('max_distance')
            self.min_distance = config.get('filter_by_distance').get('min_distance')

            self.radius_gd = config.get('filter_ground_plane').get('radius_normals')
            self.max_nn_gd = config.get('filter_ground_plane').get('maximum_neighbors')

            self.radius_normals = config.get('normals').get('radius_normals')
            self.max_nn = config.get('normals').get('maximum_neighbors')

            self.fpfh_threshold = config.get('fpfh').get('fpfh_threshold')
            self.max_radius_descriptor = config.get('descriptor').get('max_radius_descriptor')
            self.distance_threshold = config.get('icp').get('distance_threshold')

class Exp_ParametersConfig():
    """
    Clase en la que se almacenan los parametros del experimentos
    """
    def __init__(self, yaml_file='config/exp_parameters.yaml'):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)
            self.directory = config.get('directory')
            self.save_overlap_as = config.get('overlap').get('save_as')
            self.overlap_radius = config.get('overlap').get('radius')
            self.scan_idx = config.get('scan_idx')
            self.do_offline_ekf = config.get('do_offline_ekf')

            self.exp_deltaxy = config.get('experiment').get('deltaxy')
            self.exp_deltath = config.get('experiment').get('deltath')
            self.exp_long = config.get('experiment').get('long')
            self.local_dist = config.get('local_environment').get('dist')
            self.local_angle = config.get('local_environment').get('angle')

            self.gps_status = config.get('gps').get('reference_status')
            self.origin_lat = config.get('origin_coordinates').get('lat')
            self.origin_lon = config.get('origin_coordinates').get('lon')

class Debugging_ParametersConfig():
    """
    Clase en la que se almacenan los parametros del debug
    """
    def __init__(self, yaml_file='config/debugging_parameters.yaml'):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)
            self.do_debug = config.get('do_debug')
            self.load_overlap = config.get('load_saved_overlap')

            self.plot_trajectory = config.get('plot_maps').get('trajectory')
            self.plot_overlap = config.get('plot_maps').get('overlap')
            self.plot_registration_result = config.get('plot_scans').get('registration_result')
            self.plot_initial_tranform = config.get('plot_scans').get('initial_transform')
            self.plot_scan_overlap = config.get('plot_scans').get('overlap')

class Training_ParametersConfig():
    """
    Clase en la que se almacenan los parametros de entrenamiento
    """
    def __init__(self, yaml_file='config/training_parameters.yaml'):
        with open(yaml_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
            print(config)


            self.testing_path = config.get('testing_path')

            self.parallel_computing = config.get('parallel_computing')
            self.max_ngpu = config.get('max_ngpu')

            self.sample_points = config.get('input_data_preprocessing').get('sample2n')
            self.number_of_points = config.get('input_data_preprocessing').get('n_points')
            self.max_radius = config.get('input_data_preprocessing').get('max_radius')
            self.normalize_coords = config.get('input_data_preprocessing').get('normalize_coords')

            self.normalize_embeddings = config.get('output_data').get('normalize_embeddings')
            self.output_size = config.get('output_data').get('embedding_size')

            self.training_path = config.get('training_parameters').get('path')
            self.training_batch_size = config.get('training_parameters').get('batch_size')
            self.number_of_epochs = config.get('training_parameters').get('number_epochs')
            self.complete_epochs = config.get('training_parameters').get('complete_epochs')


            self.validation_path = config.get('validation_parameters').get('path')
            self.validation_batch_size = config.get('validation_parameters').get('batch_size')

            self.groundtruth_path = config.get('groundtruth_parameters').get('path')
            self.groundtruth_batch_size = config.get('groundtruth_parameters').get('batch_size')

            self.success_radius = config.get('evaluation_parameters').get('success_boundary_radius')



ICP_PARAMETERS = ICP_ParametersConfig()
EXP_PARAMETERS = Exp_ParametersConfig()
DEBUGGING_PARAMETERS = Debugging_ParametersConfig()
TRAINING_PARAMETERS = Training_ParametersConfig()
