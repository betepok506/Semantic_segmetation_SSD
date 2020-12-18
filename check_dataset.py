import os
from lxml import etree, objectify
import cv2 as cv
from PIL import Image
from keras.preprocessing import image
import numpy as np
from keras import backend as K
from models.keras_ssd300 import ssd_300
from keras.optimizers import Adam, SGD
from imageio import imread
from keras_loss_function.keras_ssd_loss import SSDLoss
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
# path_source='D:\Kyrs\ssd_300\ssd_keras\dataset\VOCdevkitnew_4classes\VOC2012\Annotations'
path_source='OpenImage_300_4classes/OpenImage/Annotations/'
dict_difficult={}
rez_dict={}
rez_in_dataset={}
area_list=[90,300,500,1100,1600,2400,3600,5400,7800,9800,13500,15600,18000,21000,25000,29900,36000,39800,44000,55000,64000,72000,80000,87000,90000,100000]
# classes = ['background','car', 'aeroplane', 'person', 'dog']
classes = ['background','Car','Airplane','Person','Dog']
# classes = ['background',
#                'aeroplane', 'bicycle', 'bird', 'boat',
#                'bottle', 'bus', 'car', 'cat',
#                'chair', 'cow', 'diningtable', 'dog',
#                'horse', 'motorbike', 'person', 'pottedplant',
#                'sheep', 'sofa', 'train', 'tvmonitor']
def parseXML(xmlFile):
    """
    Парсинг XML
    """
    bound_boxes_class={}
    with open(xmlFile) as fobj:
        xml = fobj.read()
    global dict_difficult
    root = etree.fromstring(xml)
    for appt in root.getchildren():
        if appt.tag=='filename':
            image_name =appt.text

        xmin=-1
        ymax=-1
        xmax=-1
        ymin=-1
        if appt.tag == 'object':
            name_object=''
            for elem in appt.getchildren():
                if elem.tag=='bndbox':
                    for coord in elem.getchildren():
                        if coord.tag == 'ymin' or coord.tag == 'ymax':
                            if coord.tag=='ymin':
                                ymin=int(coord.text)
                            if coord.tag=='ymax':
                                ymax=int(coord.text)

                        if coord.tag == 'xmin' or coord.tag == 'xmax':
                            if coord.tag=='xmin':
                                xmin=int(coord.text)
                            if coord.tag=='xmax':
                                xmax=int(coord.text)
                    if bound_boxes_class.get(name_object)==None:
                        bound_boxes_class.setdefault(name_object,[])
                    bound_boxes_class[name_object].append([xmin,ymin,xmax,ymax])
                if elem.tag=='name':
                    name_object=elem.text
                if elem.tag=='difficult':
                    dict_difficult[name_object+"__"+str(elem.text)]=dict_difficult.setdefault(name_object+"__"+str(elem.text),0)+1
            if rez_dict.get(name_object) == None:
                rez_in_dataset.setdefault(name_object,
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            cnt_area=(ymax-ymin)*(xmax-xmin)
            for index, area in enumerate(area_list):
                if area != 100000 and cnt_area >= area_list[index] and cnt_area < area_list[index+1]:
                    rez_in_dataset[name_object][index] += 1
                    break


    predict_boxes=predict(os.path.join('OpenImage_300_4classes/OpenImage/' + 'JPEGImages',image_name))
    fun(bound_boxes_class,predict_boxes)

def fun(bound_boxes_class,predict_bound_box):
    used_bound_box=set()

    for name_classes in bound_boxes_class.keys():
        for coord_bound_box in bound_boxes_class[name_classes]:
            for coord_predict in predict_bound_box:
                if coord_predict==[]:
                    continue
                IoU=check_bound_box(coord_bound_box,coord_predict[1:])

                if IoU[0]>=0.50 and classes[int(coord_predict[0])]==name_classes:

                    if rez_dict.get(int(coord_predict[0])) == None:
                        rez_dict.setdefault(coord_predict[0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
                    for index,area in enumerate(area_list):
                        if area!=100000 and IoU[1]>=area and IoU[1] < area_list[index+1]:
                            rez_dict[coord_predict[0]][index]+=1
                            break




def check_bound_box(bound_box,rect_pred): # bound box xml файла xmin,ymin,xmax,ymax
    IoU=0
    if bound_box[1]<= rect_pred[1]<=bound_box[3]:
        y_len=min(rect_pred[3],bound_box[3])-rect_pred[1]
        x_len=min(bound_box[2],rect_pred[2])-max(bound_box[0],rect_pred[0])
        IoU=x_len*y_len
    elif bound_box[1] <=rect_pred[3] <= bound_box[3]:
        y_len=rect_pred[3]-max(rect_pred[1],bound_box[1])
        x_len=min(bound_box[2],rect_pred[2])-max(bound_box[0],rect_pred[0])
        IoU = x_len * y_len
    elif rect_pred[1]<=bound_box[1]<=rect_pred[3]:
        y_len=min(rect_pred[3],bound_box[3]) - max(rect_pred[1],bound_box[1])
        x_len= min(rect_pred[2],bound_box[2]) - max(rect_pred[0],bound_box[0])
        IoU = x_len * y_len
    if IoU<0:
        IoU=0;
    # area=(((bound_box[3]-bound_box[1])*(bound_box[2]-bound_box[0])) + ((rect_pred[3]-rect_pred[1])*(rect_pred[2]-rect_pred[0])) - IoU)
    return (IoU/(((bound_box[3]-bound_box[1])*(bound_box[2]-bound_box[0])) + ((rect_pred[3]-rect_pred[1])*(rect_pred[2]-rect_pred[0])) - IoU),IoU)



def predict(path_predict):
    img_height = 300
    img_width = 300

    orig_images = []  # Store the images here.
    input_images = []  # Store resized versions of the images here.

    img_path = path_predict ###########################################

    orig_images.append(imread(img_path))
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    y_pred = model.predict(input_images)

    confidence_threshold = 0.5

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    # print("Predicted boxes:\n")
    # print('   class   conf xmin   ymin   xmax   ymax')
    # print(y_pred_thresh[0])
    # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # classes = ['background','car', 'aeroplane', 'person', 'dog']

    rez=[]
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        xmin = box[2] * orig_images[0].shape[1] / img_width
        ymin = box[3] * orig_images[0].shape[0] / img_height
        xmax = box[4] * orig_images[0].shape[1] / img_width
        ymax = box[5] * orig_images[0].shape[0] / img_height
        rez.append((box[0],xmin,ymin,xmax,ymax))
        # color = colors[int(box[0])]
        # label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        # current_axis.add_patch(
        #     plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
        # current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor': color, 'alpha': 1.0})
    return rez
    # mean_average_precision, average_precisions, precisions, recalls = results
    # plt.show()

def model_compile():
    img_height = 300
    img_width = 300
    global model
    K.clear_session()  # Clear previous models from memory.

    model = ssd_300(image_size=(img_height, img_width, 3),
                    n_classes=4,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05],
                    # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 100, 300],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

    # 2: Load the trained weights into the model.

    # TODO: Set the path of the trained weights.
    # weights_path = 'VGG_VOC0712_SSD_300x300_iter_120000.h5' #########
    # weights_path = '../save/ssd300_pascal_07+12_BEST_300_4.h5'
    # weights_path='../VGG_VOC0712_SSD_300x300_iter_120000.h5'
    weights_path='..\\save_OpenIm\\300_4\\OpenIm_300epoch-60_loss-4.7389_val_loss-4.6681.h5'
    model.load_weights(weights_path, by_name=True)

    # 3: Compile the model so that Keras won't complain the next time you load it.

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

if __name__=="__main__":
    # print(check_bound_box([1,1,5,4],[4,3,8,5]))
    # print(check_bound_box([1, 1, 5, 4], [5, 4, 8, 5]))
    # print(check_bound_box([1, 1, 5, 4], [2, 2, 4, 3]))
    # print(check_bound_box([1, 1, 5, 4], [0, 0, 6, 5]))
    # print(check_bound_box([1, 1, 5, 4], [0, 0, 3, 5]))
    model_compile()
    for file_xml in os.listdir(path_source):
        parseXML(os.path.join(path_source,file_xml))

    # for name_classes in rez_dict.keys():
    #     print(f'Name classes {classes[int(name_classes)]} ----- ')
    #     for index,size in enumerate(rez_dict[name_classes]):
    #         print(f'\t {area_list[int(index)]} -- {size}')
    #     print('--------------------------')

    print("0============================")
    for name_classes in rez_in_dataset.keys():
        print(f'Name classes {name_classes} ----- ')
        for index,size in enumerate(rez_in_dataset[name_classes]):
            print(f'\t {area_list[int(index)]} -- {size}')
        print('--------------------------')
    # print(dict_difficult)