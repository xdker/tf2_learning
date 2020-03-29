from tensorflow.keras.optimizers import RMSprop
from transfer_learning import DenseLearner, LossHistory, TensorBoard
from visualization import training_vis

train = "data/animals/train"
val = "data/animals/val"
size = (224, 224)
batch_size = 32
optimizer = RMSprop(lr=1e-4)

learner = DenseLearner(objective='categorical', base='resnet50', layers=[512, 128], dropout=[0.5, 0.5])

tbCallBack = TensorBoard(log_dir="./logs", histogram_freq=1, write_grads=True, write_images=True, update_freq='batch')
logs1 = LossHistory()

history = learner.flow_from_directory(train, val, optimizer=optimizer, epochs=80, custom_callbacks=[tbCallBack, logs1])
# training_vis(history)
logs1.loss_plot()
learner.save('resnet_base.pickle')
learner_2 = DenseLearner(objective='categorical', base='resnet50', layers=[512, 256], dropout=[0.5, 0.5])
# unfreeze 5 last convolutional layers of the base network and fit the model again
tbCallBack2 = TensorBoard(log_dir="./logs", histogram_freq=1, write_grads=True, write_images=True, update_freq='batch')
logs2 = LossHistory()
history = learner_2.flow_from_directory(train, val, optimizer=optimizer, epochs=15,
                                        custom_callbacks=[tbCallBack2, logs2], unfreeze=5)

learner_2.save('resnet_base_unfreeze.pickle')

