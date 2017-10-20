# Neural Belief Tracker

Contact: Nikola Mrkšić (nikola.mrksic@gmail.com)

An implementation of the Fully Data-Driven version of the Neural Belief Tracking (NBT) model. 

This version of the model uses a learned belief state update in place of the rule-based mechanism used in the original paper.  

### Configuring the Tool

The config file in the config directory specifies the model hyperparameters, training details, dataset, ontologies, etc. 

### Running Experiments

train.sh and test.sh can be used to train and test the model (using the default config file). 
track.sh uses the trained models to 'simulate' a conversation where the developer can enter sequential user turns and observe the change in belief state.  


