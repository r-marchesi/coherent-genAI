modalities_list = ['cna','rnaseq','rppa', 'wsi']
data_dir = '../datasets_TCGA/07_normalized/'

simple_exps_1 = [('cna', 'rnaseq'), ('cna', 'rppa'), ('cna', 'wsi'), ('rnaseq', 'cna'), ('rnaseq', 'rppa'), ('rnaseq', 'wsi')] 
simple_exps_2 = [('rppa', 'cna'), ('rppa', 'rnaseq'), ('rppa', 'wsi'), ('wsi', 'cna'), ('wsi', 'rnaseq'), ('wsi', 'rppa')]
multi_rnaseq_exp = ['rnaseq_from_multi']


params = {

# architecture
'architecture' : ['norm_mlp'], 

# training params
'batch_size' : [64, 128],
'num_epochs' : [20000],
'learning_rate' : [1e-4, 1e-3],
'validation_epochs': [200], # test model on validation set every n epochs
'max_iter' : [10],

# model params
'initial_size' : [256, 512, 1024],
'bottleneck_size' : [8, 16, 32], 
'n_layers' : [4,5,6,7],
'time_embedding_dimension' : [64, 128], 
'cond_embedding_dim' : [32]
}
