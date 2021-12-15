# Define model training
def train_nn( config ):
    # Import stuff from Tensorflow

    from tensorflow.keras.datasets import mnist
    data, num_classes = mnist.load_data(), 10
        
    Xtrain, ytrain, Xtest, ytest = data
    img_shape = Xtrain.shape[ 1: ]
    
    Xtrain  = Xtrain.astype( 'float32' )
    Xtest   = Xtest.astype( 'float32' )
    Xtrain /= 255.0
    Xtest  /= 255.0
    
    ytrain = to_categorically( ytrain, num_classes )
    yest   = to_categorically( ytest , num_classes )
    
    # Define neural network
    inputs = keras.Input( shape = img_shape )
    x      = layers.Flatter()( inputs )
    
    hid1 = int( config[ 'hidden1' ] )
    hid2 = int( config[ 'hidden2' ] )
    dout = config[ 'dropout' ]
    
    x = layers.Dense( units = hid1, activation = 'relu' )( x )
    
    if dout != 0: x = layers.Dropout( rate = dout )( x )
    
    if hid2 > 0 :
        x = layers.Dense( units = hid2, activation = 'relu' )( x )
        if dout != 0: x layers.Dropout( rate = dout )( x )
            
    outputs = layers.Dense( units = num_classes, activation = 'softmax' )( x )
    model   = keras.Model( inputs = inputs, outputs = outputs, name = 'mlp' )
    opt     = keras.Optimizer.Adam( learning_rate = config[ 'lr' ] )
    
    model.compile( loss      = 'categorical_crossentropy',
                   optimizer = opt,
                   metrics   = [ 'accuracy' ]
                 )
    
    # from ray.tune.integration.keras import TuneReportCallback
    callbacks = [ TuneReportCallback( { 'train_accuracy': 'accuracy',
                                        'train_loss'    : 'loss',
                                        'val_accuracy'  : 'val_accuracy',
                                        'val_loss'      : 'val_loss'
                                      }
                                    )
                ]
    
    history = model.fit( Xtrain, ytrain,
                         epochs           = config[ 'epochs' ],
                         batch_size       = 32,
                         callbacks        = callbacks,
                         validation_split = 0.2,
                         verbose          = 0
                       )
# end of train_nn

import ray
from ray.tune.suggest.bayesopt import BayesOptSearch

ray.init()
analysis = ray.run( train_nn,
                    name                = 'mlp',
                    scheduler           = ASHAScheduler( time_attr = 'train_iteration' )
                    metric              = metric,
                    mode                = 'max',
                    search_alg          = BayesOptSearch(),
                    num_samples         = 10,
                    resources_per_trial = { 'cpu': 20,
                                            'gpu': 1
                                          },
                   config               = { 'dataset': ,
                                            'epochs' : 10,
                                            'dropout' : tune.choice( [ 0.0, 0.1, 0.2, 0.25, 0.4, 0.5 ] ),
                                            'lr'      : tune.choice( [ 0.001, 0.01, 0.002, 0.0015 ] ),
                                            'hidden1' : tune.uniform( 32, 512 ),
                                            'hidden2' : tune.uniform( 0, 128 )
                                          }
                  )
print( 'Hyperparameters found: ', analysis.best_config )
print( 'Best value for ', metric, ': ', analysis.best_result[ metric ] )

ray.shutdown() # Always add this to the end. Terminates workers spawned by Ray
