from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator



def Data():
    (x_train, _), (_, _ )= cifar10.load_data()
    datagen = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)

    return datagen.fit(x_train)






