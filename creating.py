from keras.models import Sequential
from keras.layers import Dense,Convolution2D,Flatten,MaxPooling2D

clf=Sequential()
clf.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#convolution@D(no. feature detectors,rows,columns), input_shape=(inp_row,inp_col,RGB=3)
clf.add(MaxPooling2D(pool_size = (2, 2)))
clf.add(Convolution2D(32, (3, 3), activation = 'relu'))
clf.add(MaxPooling2D(pool_size = (2, 2)))
clf.add(Dense(units = 128, activation = 'relu'))
clf.add(Dense(units = 1, activation = 'sigmoid'))
clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.fit_generator(training_set,
#                         steps_per_epoch = 8000,
 #                        epochs = 25,
  #                       validation_data = test_set,
   #                      validation_steps = 2000)
   
