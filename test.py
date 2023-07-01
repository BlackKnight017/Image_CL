from PIL import ImageTk, Image
import numpy

from keras.models import load_model
model = load_model('image_CL.h5')

classes = { 
    0:'aeroplane',
    1:'automobile',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    6:'frog',
    7:'horse',
    8:'ship',
    9:'truck' 
}

image = Image.open("cat1.png")
image = image.resize((32,32))
image = numpy.expand_dims(image, axis=0)
image = numpy.array(image)

pred = model.predict(image)[0]

max_index = numpy.argmax(pred)

print(classes[max_index])

# print(model.predict([image])[0][4])
