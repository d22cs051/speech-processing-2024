class Config:
    def __init__(self):
        self.custom_data_dir = 'data/Dataset_Speech_Assignment'
        self.for2sec_data_dir = 'data/for-2seconds'
        self.batch_size = 32
        self.num_workers = 4
        self.num_epochs = 50
        self.lr = 1e-3
        self.model_checkpoint_path = 'models/Best_LA_model_for_DF.pth'
        
        ############################################################################
        """    
        parser.add_argument('--algo', type=int, default=3, 
                        help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                            5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .default=0]')

        # LnL_convolutive_noise parameters 
        parser.add_argument('--nBands', type=int, default=5, 
                        help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
        parser.add_argument('--minF', type=int, default=20, 
                        help='minimum centre frequency [Hz] of notch filter.[default=20] ')
        parser.add_argument('--maxF', type=int, default=8000, 
                        help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
        parser.add_argument('--minBW', type=int, default=100, 
                        help='minimum width [Hz] of filter.[default=100] ')
        parser.add_argument('--maxBW', type=int, default=1000, 
                        help='maximum width [Hz] of filter.[default=1000] ')
        parser.add_argument('--minCoeff', type=int, default=10, 
                        help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
        parser.add_argument('--maxCoeff', type=int, default=100, 
                        help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
        parser.add_argument('--minG', type=int, default=0, 
                        help='minimum gain factor of linear component.[default=0]')
        parser.add_argument('--maxG', type=int, default=0, 
                        help='maximum gain factor of linear component.[default=0]')
        parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                        help=' minimum gain difference between linear and non-linear components.[default=5]')
        parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                        help=' maximum gain difference between linear and non-linear components.[default=20]')
        parser.add_argument('--N_f', type=int, default=5, 
                        help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

        # ISD_additive_noise parameters
        parser.add_argument('--P', type=int, default=10, 
                        help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
        parser.add_argument('--g_sd', type=int, default=2, 
                        help='gain parameters > 0. [default=2]')

        # SSI_additive_noise parameters
        parser.add_argument('--SNRmin', type=int, default=10, 
                        help='Minimum SNR value for coloured additive noise.[defaul=10]')
        parser.add_argument('--SNRmax', type=int, default=40, 
                        help='Maximum SNR value for coloured additive noise.[defaul=40]')
        """
        ############################################################################
        # conversion from agrparse to class object
        self.algo = 3
        self.nBands = 5
        self.minF = 20
        self.maxF = 8000
        self.minBW = 100
        self.maxBW = 1000
        self.minCoeff = 10
        self.maxCoeff = 100
        self.minG = 0
        self.maxG = 0
        self.minBiasLinNonLin = 5
        self.maxBiasLinNonLin = 20
        self.N_f = 5
        self.P = 10
        self.g_sd = 2
        self.SNRmin = 10
        self.SNRmax = 40
        
        
        #############################################################################
        """
        parser.add_argument('--database_path', type=str, default='/your/path/to/data/ASVspoof_database/DF/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 DF for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 DF eval data folders are in the same database_path directory.') 
        '''
        % database_path/
        %   |- DF
        %      |- ASVspoof2021_DF_eval/flac
        %      |- ASVspoof2019_LA_train/flac
        %      |- ASVspoof2019_LA_dev/flac
        '''

        parser.add_argument('--protocols_path', type=str, default='database/', help='Change with path to user\'s DF database protocols directory address')
        '''
        % protocols_path/
        %   |- ASVspoof_LA_cm_protocols
        %      |- ASVspoof2021.LA.cm.eval.trl.txt
        %      |- ASVspoof2019.LA.cm.dev.trl.txt 
        %      |- ASVspoof2019.LA.cm.train.trn.txt
    
        %   |- ASVspoof_DF_cm_protocols
        %      |- ASVspoof2021.DF.cm.eval.trl.txt
    
        '''

        # Hyperparameters
        parser.add_argument('--batch_size', type=int, default=14)
        parser.add_argument('--num_epochs', type=int, default=100)
        parser.add_argument('--lr', type=float, default=0.000001)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument('--loss', type=str, default='weighted_CCE')
        # model
        parser.add_argument('--seed', type=int, default=1234, 
                            help='random seed (default: 1234)')
        
        parser.add_argument('--model_path', type=str,
                            default=None, help='Model checkpoint')
        parser.add_argument('--comment', type=str, default=None,
                            help='Comment to describe the saved model')
        # Auxiliary arguments
        parser.add_argument('--track', type=str, default='DF',choices=['LA', 'PA','DF'], help='LA/PA/DF')
        parser.add_argument('--eval_output', type=str, default=None,
                            help='Path to save the evaluation result')
        parser.add_argument('--eval', action='store_true', default=False,
                            help='eval mode')
        parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
        parser.add_argument('--eval_part', type=int, default=0)
        # backend options
        parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                            default=True, 
                            help='use cudnn-deterministic? (default true)')    
        
        parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                            default=False, 
                            help='use cudnn-benchmark? (default false)') 
        """
        
        self.weight_decay = 0.0001
        self.loss = 'weighted_CCE'
        self.seed = 1234
        self.model_path = "models/LA_model.pth"
        self.comment = None
        self.track = 'DF'
        self.eval_output = None
        self.eval = False
        self.is_eval = False
        self.eval_part = 0
        self.cudnn_deterministic_toggle = False
        self.cudnn_benchmark_toggle = False
        
        self.wandb_config = {
            'project': 'Speech Assignment 3',
            'run_name': 'LA_model',
        }
        