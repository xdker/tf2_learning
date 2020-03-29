from transfer_learning import DenseLearner

learner = DenseLearner.load('logs/resnet50.pickle')
print(learner.explain_prediction("data/animals/train/antelope/Img-10000.jpg"))
