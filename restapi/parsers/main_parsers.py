import camelot
import numpy as np
import tensorflow as tf
from PIL import Image
from pytesseract import pytesseract
from tensorflow.keras.layers import Dropout, Input, Conv2DTranspose, concatenate, UpSampling2D,Conv2D
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Input, Conv2DTranspose, concatenate, UpSampling2D,Conv2D
import os
from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
PDFInfoNotInstalledError,
PDFPageCountError,
PDFSyntaxError)

pytesseract.tesseract_cmd = r"C:\Practice\Python\Django\cringe\tesseract\tesseract.exe"

class tbl_decoder(tf.keras.layers.Layer):
  def __init__(self, name = "Table_mask"):
    super().__init__(name = name)
    self.conv1 = Conv2D(filters=512, kernel_size=(1,1), activation='relu')
    self.umsample1 = UpSampling2D(size = (2,2),)
    self.umsample2 = UpSampling2D(size = (2,2),)
    self.umsample3 = UpSampling2D(size = (2,2),)
    self.umsample4 = UpSampling2D(size = (2,2),)
    self.convtranspose = Conv2DTranspose( filters=3, kernel_size=3, strides=2, padding = 'same')

  def call(self, X):
    input,pool_3,pool_4 = X[0],X[1],X[2]
    x = self.conv1(input)
    x = self.umsample1(x)
    x = concatenate([x, pool_4])
    x = self.umsample2(x)
    x = concatenate([x, pool_3])
    x = self.umsample3(x)
    x = self.umsample4(x)
    x = self.convtranspose(x)
    return x

class col_decoder(tf.keras.layers.Layer):
  def __init__(self, name = "Column_mask"):
    super().__init__(name = name)
    self.conv1 = Conv2D(filters=512, kernel_size=(1,1), activation='relu')
    self.drop  = Dropout(0.8)
    self.conv2 = Conv2D(filters=512, kernel_size=(1,1), activation='relu')
    self.umsample1 = UpSampling2D(size = (2,2),)
    self.umsample2 = UpSampling2D(size = (2,2),)
    self.umsample3 = UpSampling2D(size = (2,2),)
    self.umsample4 = UpSampling2D(size = (2,2),)
    self.convtranspose = Conv2DTranspose( filters=3, kernel_size=3, strides=2, padding = 'same')

  def call(self, X):
    input,pool_3,pool_4 = X[0],X[1],X[2]
    x = self.conv1(input)
    x = self.drop(x)
    x = self.conv2(x)
    x = self.umsample1(x)
    x = concatenate([x, pool_4])
    x = self.umsample2(x)
    x = concatenate([x, pool_3])
    x = self.umsample3(x)
    x = self.umsample4(x)
    x = self.convtranspose(x)
    return x


def table_detection(path) :
  image = tf.io.read_file(path)
  image = tf.image.decode_image(image, channels=3, expand_animations = False)
  image = tf.image.resize(image, [1024, 1024])
  image = tf.cast(image, tf.float32) / 255.0

  mask1= model.predict(image[np.newaxis, :, :, :])[0]

  table_mask= get_mask(mask1)
  table_mask = tf.keras.preprocessing.image.array_to_img(table_mask)
  table_mask = table_mask.resize((1024, 1024), Image.ANTIALIAS)


  img_org = tf.keras.preprocessing.image.array_to_img(image)
  img_org = img_org.resize((1024, 1024), Image.ANTIALIAS)

  img_mask = table_mask.convert('L')

  img_org.putalpha(img_mask)

  return img_org

def get_text(img_org):
  text_list = pytesseract.image_to_string(img_org, lang='rus')
  text_list = text_list.split('\n')
  while("" in text_list)  :
    text_list.remove("")
  while(" " in text_list)  :
    text_list.remove(" ")
  while("  " in text_list) :
    text_list.remove("  ")
    
  res = []
  for val in text_list:
    pars = val.split(" ")
    res.append([pars[0],pars[1], pars[2]])

  return res

def get_mask(mask):
  mask = tf.argmax(mask, axis=-1)
  mask = mask[..., tf.newaxis]
  return mask[0]


input = Input(shape=(1024,1024,3))
vgg19 = tf.keras.applications.VGG19(include_top=False, weights = 'imagenet',
                                    input_tensor=input, classes= 1000)

x = vgg19.output
x = Conv2D(512, (1, 1), activation = 'relu', name='block6_conv1')(x)
x = Dropout(0.8, name='block6_dropout1')(x)
x = Conv2D(512, (1, 1), activation = 'relu', name='block6_conv2')(x)
x = Dropout(0.8, name = 'block6_dropout2')(x)

Table_Decoder  = tbl_decoder()
Column_Decoder = col_decoder()

pool_3 = vgg19.get_layer('block3_pool').output
pool_4 = vgg19.get_layer('block4_pool').output

output1 = Table_Decoder([x, pool_3, pool_4])
output2 = Column_Decoder([x, pool_3, pool_4])

model = Model(inputs = input, outputs= [output1,output2], name = "TableNet")
model.load_weights(r"C:\Practice\Python\bin\files\weights-04-0.1819--.hdf5")

class Analyze():
  def __init__(self, name, value, units):
        self.name = name
        self.value = value
        self.units = units


dataset = {
    'гемоглобин hgb': ['г/л', 'g/L', 'г/дл'],
    'эритроциты rbc': ['10^12/л', '10E12/L', 'млн/мкл'],
    'средний объем эритроцитов mcv': ['фемтолитр', 'fl', 'фл'],
    'среднее содержание гемоглобина в эритроците mchc': ['пкг', 'pg'],
    'тромбоциты plt': ['10^9/л', '10E9/L', 'тыс/мкл', 'x10^9/л'],
    'лейкоциты wbc': ['10^9/л', '10E9/L', 'тыс/мкл', 'x10^9/л'],
    'гематокрит hct': ['%'],
    'скорость оседания эритроцитов соэ':['мм/ч'],
    'нормобласты, %': ['%'],
    'базофилы, balso %': ['%'],
    'эозинофилы, абс. eos eo':['тыс/мкл ', '10^9/л'],
    'нейтрофилы, абс. neut ne':['%', '10^9/л', '10E9/L', 'тыс/мкл', 'x10^9/л'],
    'лимфоциты, % lymph lymf' : ['%'],
    'моноциты, % mon Mono':['%'],
}

def parse_pdf(directory):
    table = camelot.read_pdf(directory)

    if len(table) > 0:
      table_list = table[0].df.values.tolist()

      analyzes =[]
      for row in table_list:
        for item in row:
            for key in  dataset.keys():
                  if(item.lower() in key):
                    for i in row:
                      if(i.lower() in dataset[key]):
                        units = i
                        value = row[row.index(i)-1]
                        analyzes.append(Analyze(item.lower(), value, units))
                        break
                    break
            break  

      result = list()

      for i in analyzes:
        result.append([i.name, i.value, i.units])
              
      return result

    else:
      try:
        images = convert_from_path(directory)
        for image in images:
          image.save(os.path.splitext(directory)[0]+'.jpg', "JPEG")
        return get_text(table_detection(os.path.splitext(directory)[0]+'.jpg'))
      except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
        return ["Файл не был распознан"]
        
