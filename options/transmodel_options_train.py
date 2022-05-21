
from .baseoptions import BaseTrainOptions

class TrainOptions(BaseTrainOptions):
    def init_options(self):
        
        data = self.parser.add_argument_group('configurations related to data')
        data.add_argument('--data_path', type=str, default='data/my-1230-honey-rot-processed', help='data dir')
        data.add_argument('--evaldata_path', type=str, default='data/my-1230-honey-rot-processed', help='data dir')
        data.add_argument('--data_type', type=str, choices=['blender', 'splishsplash', 'GT'])
        
        train = self.parser.add_argument_group('Training configuration')
        train.add_argument('--pretrained_transition_model', type=str, default=None)
        train.add_argument('--transition_lr', type=float, default=1e-8, help='initial learning rate')
        train.add_argument('--N_iters', type=int, default=100000, help='Total iteration steps')
        train.add_argument('--seed', type=int, default=10, help='random seed')
        train.add_argument('--grad_clip_value', type=float, default=0, help='0: no gradient clip')

        transition = self.parser.add_argument_group('Training Renderer configuration')
        transition.add_argument('--gravity', type=str, default='0,0,-9.81')
        
        redundant_param = self.parser.add_argument_group('redundant parameters')
        redundant_param.add_argument('--scale', type=int, default=1)
        redundant_param.add_argument('--imgW', type=int, default=400, help='image width')
        redundant_param.add_argument('--imgH', type=int, default=400, help='image height')
        redundant_param.add_argument('--viewnames', type=str, default='view_1,view_2', help='view1,view2')
        redundant_param.add_argument('--test_viewnames', type=str, default='view_1,view_2', help='view1,view2')
        redundant_param.add_argument('--start_index', type=int, default=0)
        redundant_param.add_argument('--end_index', type=int, default=50)
       
        
