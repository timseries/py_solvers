[Application1]
Name=blurredCameraman

[Input1]
Name=Input
FileDir=~/GoogleDrive/timothy.daniel.roberts@gmail.com/PhD/Projects/Classification/Data/GalaxyClassification/images_training_rev1_formatted
FileMember=ids
FileName=class_csv_exp1.csv

[Input2]
Name=Input
FileDir=~/GoogleDrive/timothy.daniel.roberts@gmail.com/PhD/Projects/Classification/Data/GalaxyClassification/images_training_rev1_formatted
FileMember=
FileName=allfeatures.pkl

[Preprocess1]
Name=Preprocess

[Observe1]
Name=Observe
ObservationType=classification
Transforms=TransformArray2
TrainingProportion=0.1
Shuffle=0
ShuffleSeed=0

[Transform1]
Name=DTCWT
nlevels=4
#biort=near_sym_b
#qshift=qshift_b
biort=near_sym_b_bp
qshift=qshift_b_bp
OpenCL=1

[Transform2]
Name=Scattering
Depth=3
Transform=Transform1

[TransformArray1]
Name=OperatorComp
Operators=Transform1

[TransformArray2]
Name=OperatorComp
Operators=Transform2

[Solver1]
Name=Classify
Transforms=TransformArray2
FeatureReduction=average
#FeatureSectionInput=Input2
FeatureSectionOutput=OutputObject1
#Method=svc
# Method=linearsvc
Method=affinepca
Results=Results1
PreprocessMethod=normalize
#svm parameters
kwsvc_c=1.0
kwsvc_dual=0
kwsvc_fit_intercept=1
kwsvc_intercept_scaling=1
kwsvc_loss=l2
kwsvc_multi_class=ovr
kwsvc_penalty=l1
kwsvc_probability=1
kwsvc_random_state=0
kwsvc_tol=0.0001
kwsvc_verbose=0
#pca parameters
kwpca_n_components=60

[Results1]
Name=Results
Metrics=ScalarOutput1.1,ScalarOutput1.2,ScalarOutput1.3
#Metrics=ScalarOutput2.1,ScalarOutput2.2
#Metrics=ScalarOutput3.1,ScalarOutput3.2
#Metrics=ScalarOutput4.1,ScalarOutput4.2
#Metrics=ScalarOutput5.1,ScalarOutput5.2,ScalarOutput5.3,ScalarOutput5.4
#Metrics=ScalarOutput6.1,ScalarOutput6.2
#Metrics=ScalarOutput7.1,ScalarOutput7.2,ScalarOutput7.3
#Metrics=ScalarOutput8.1,ScalarOutput8.2,ScalarOutput8.3,ScalarOutput8.4,ScalarOutput8.5,ScalarOutput8.6,ScalarOutput8.7
#Metrics=ScalarOutput9.1,ScalarOutput9.2,ScalarOutput9.3
#Metrics=ScalarOutput10.1,ScalarOutput10.2,ScalarOutput10.3
#Metrics=ScalarOutput11.1,ScalarOutput11.2,ScalarOutput11.3,ScalarOutput11.4,ScalarOutput11.5,ScalarOutput11.6
Desktop=1
FigureGridWidth=3
FigureGridHeight=2
OutputDirectory=~/repos/latex/+scattering/
# OutputFilename=linearsvc_scattersum_l1_J5
OutputFilename=affinepca
OverwriteResults=0
RowOffset=0

[ScalarOutput1.1]
Name=Scalar
Title=Y Prediction
Key=Class1.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput1.2]
Name=Scalar
Title=Y Prediction
Key=Class1.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput1.3]
Name=Scalar
Title=Y Prediction
Key=Class1.3
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput2.1]
Name=Scalar
Title=Y Prediction
Key=Class2.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput2.2]
Name=Scalar
Title=Y Prediction
Key=Class2.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput3.1]
Name=Scalar
Title=Y Prediction
Key=Class3.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput3.2]
Name=Scalar
Title=Y Prediction
Key=Class3.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput4.1]
Name=Scalar
Title=Y Prediction
Key=Class4.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput4.2]
Name=Scalar
Title=Y Prediction
Key=Class4.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput5.1]
Name=Scalar
Title=Y Prediction
Key=Class5.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput5.2]
Name=Scalar
Title=Y Prediction
Key=Class5.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput5.3]
Name=Scalar
Title=Y Prediction
Key=Class5.3
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput5.4]
Name=Scalar
Title=Y Prediction
Key=Class5.4
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput6.1]
Name=Scalar
Title=Y Prediction
Key=Class6.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput6.2]
Name=Scalar
Title=Y Prediction
Key=Class6.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput7.1]
Name=Scalar
Title=Y Prediction
Key=Class7.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput7.2]
Name=Scalar
Title=Y Prediction
Key=Class7.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput7.3]
Name=Scalar
Title=Y Prediction
Key=Class7.3
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput8.1]
Name=Scalar
Title=Y Prediction
Key=Class8.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput8.2]
Name=Scalar
Title=Y Prediction
Key=Class8.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput8.3]
Name=Scalar
Title=Y Prediction
Key=Class8.3
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput8.4]
Name=Scalar
Title=Y Prediction
Key=Class8.4
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput8.5]
Name=Scalar
Title=Y Prediction
Key=Class8.5
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput8.6]
Name=Scalar
Title=Y Prediction
Key=Class8.6
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput8.7]
Name=Scalar
Title=Y Prediction
Key=Class8.7
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput9.1]
Name=Scalar
Title=Y Prediction
Key=Class9.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput9.2]
Name=Scalar
Title=Y Prediction
Key=Class9.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput9.3]
Name=Scalar
Title=Y Prediction
Key=Class9.3
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput10.1]
Name=Scalar
Title=Y Prediction
Key=Class10.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput10.2]
Name=Scalar
Title=Y Prediction
Key=Class10.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput10.3]
Name=Scalar
Title=Y Prediction
Key=Class10.3
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput11.1]
Name=Scalar
Title=Y Prediction
Key=Class11.1
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput11.2]
Name=Scalar
Title=Y Prediction
Key=Class11.2
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput11.3]
Name=Scalar
Title=Y Prediction
Key=Class11.3
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput11.4]
Name=Scalar
Title=Y Prediction
Key=Class11.4
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput11.5]
Name=Scalar
Title=Y Prediction
Key=Class11.5
YLabel=Class
Print=0
FigureLocation=0

[ScalarOutput11.6]
Name=Scalar
Title=Y Prediction
Key=Class11.6
YLabel=Class
Print=0
FigureLocation=0
