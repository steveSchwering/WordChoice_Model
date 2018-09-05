Here you will find models sorted by 'type' and the tools to analyze those files

Folders:
'ModelTypes' contains all kinds of models used to compare performance using various hypoerparameters and experimental parameters. For example, the 'Singleton_Ambiguous' models are each trained to produce 'ambiguous' words with various hyperparameters.
'LossAnalysis' contains files of model losses in a human-readable format and for analysis in R
'PredAnalysis' contains files of model predictions in a human-readable format and for analysis in R

Files:
'extractLosses.py' extracts model losses from each set of models in ModelTypes and organizes them in the folder 'LossAnalysis'
'extractPredictions.py' rebuilds models in ModelTypes and generates model predictions at different training stages. The predictions are then organized in folder 'PredAnalysis'
'helperfunctions_WCM.py' contains verious useful functions with some overlap of files in folder 'ModelTypes'