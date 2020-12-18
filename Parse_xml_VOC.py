from lxml import etree, objectify
import os
import shutil
import cv2 as cv
from PIL import Image
import sys

path_save='D:\\Kyrs\\ssd_300\\ssd_keras\\dataset'
path_folder='D:\Kyrs\ssd_300\ssd_keras\dataset\VOCdevkit'
# path_folder_annotation="D:\Kyrs\ssd_300\ssd_keras\dataset\VOCdevkit\VOC2007\Annotations"
path_dataset='VOCdevkitnew_4classes_512'

classes = {'car','aeroplane','person','dog'}
size=512,512
def parseXML(xmlFile,folder):
    """
    Парсинг XML
    """
    with open(xmlFile) as fobj:
        xml = fobj.read()

    root = etree.fromstring(xml)
    image_name = 'None'
    fl_save_xml = False
    width = 0
    height = 0

    for appt in root.getchildren():
        if appt.tag=='size':
            for elem in appt.getchildren():
                if elem.tag=='width':
                    width=int(elem.text)
                    elem.text=f'{size[0]}'
                if elem.tag=='height':
                    height=int(elem.text)
                    elem.text=f'{size[1]}'

    koef_y = size[0] / width
    koef_x = size[1] / height
    for appt in root.getchildren():
        if appt.tag=='filename':
            image_name =appt.text
        delete_class=False
        if appt.tag == 'object':
            for elem in appt.getchildren():
                if elem.tag=='name' and not(elem.text in classes):
                    delete_class=True
                    break

                if elem.text in classes:
                    fl_save_xml=True

                if elem.tag=='bndbox':
                    # if(width==0 or height==0):
                    #     # fl_save_xml=False
                    #     # delete_class = True
                    #     print(xmlFile)
                    #     break

                    for coord in elem.getchildren():
                        if coord.tag == 'ymin' or coord.tag == 'ymax':
                            try:
                                coord.text = f'{int(int(float(coord.text) * koef_x))}'
                            except:
                                print(xmlFile)
                        if coord.tag == 'xmin' or coord.tag == 'xmax':
                            try:
                                coord.text = f'{int(int(float(coord.text) * koef_y))}'
                            except:
                                print(xmlFile)
            if delete_class==True:
                root.remove(appt)

    if fl_save_xml==True:
        obj_xml = etree.tostring(root, pretty_print=True,
                                 xml_declaration=False)

        try:
            with open(f"{os.path.join(path_save,path_dataset,folder,'Annotations',os.path.basename(xmlFile))}", "wb") as xml_writer:
                xml_writer.write(obj_xml)
        except IOError:
            pass

    else:
        image_name = 'None'
    return image_name

def check():
    path_res_img='D:\Kyrs\ssd_300\ssd_keras\dataset\VOCdevkitnew_4classes\VOC2012\JPEGImages'
    path_res_xml='D:\Kyrs\ssd_300\ssd_keras\dataset\VOCdevkitnew_4classes\VOC2012\Annotations'

    path_save = 'D:\Kyrs\ssd_300\ssd_keras\dataset\VOCdevkitnew_4classes\VOC2012\\check'
    for file_xml in os.listdir(path_res_xml):
        with open(os.path.join(path_res_xml,file_xml)) as fobj:
            xml = fobj.read()

        root = etree.fromstring(xml)
        for img_file in os.listdir(path_res_img):
            if img_file.split('.')[0]==file_xml.split('.')[0]:
                img=cv.imread(path_res_img+'\\'+file_xml.split('.')[0]+'.jpg')
        for appt in root.getchildren():
            if appt.tag=='object':
                for elem in appt.getchildren():
                    if appt.tag=='object':
                        if elem.tag=='bndbox':
                            for coord in elem.getchildren():
                                # coord.text = f'{int(300/((width-height)*(int(coord.text)-height)))}'
                                if coord.tag=='ymin':
                                    y_min=int(coord.text)
                                elif coord.tag=='ymax':
                                    y_max=int(coord.text)
                                elif coord.tag=='xmin':
                                    x_min=int(coord.text)
                                elif coord.tag=='xmax':
                                    x_max=int(coord.text)
                            img=cv.rectangle(img, (x_min,y_min), (x_max,y_max), (0,255,0), 2)

        cv.imwrite(f"{os.path.join(path_save,file_xml.split('.')[0]+'.jpg')}",img)

def resize_image(path_sourse,path_dest):
    for img_file in os.listdir(path_sourse):
        original_image = Image.open(os.path.join(path_sourse,img_file))
        resized_image = original_image.resize(size)
        resized_image.save(os.path.join(path_dest,img_file))
        os.remove(os.path.join(path_sourse,img_file))

def create_folder(name_folder):
    try:
        os.mkdir(os.path.join(path_save,path_dataset))
    except:
        pass
    try:
        os.mkdir(os.path.join(os.path.join(path_save,path_dataset), name_folder))
    except:
        pass
    try:
        os.mkdir(os.path.join(path_save,path_dataset,name_folder,'ImageSets'))
    except:
        pass
    # try:
    #     os.mkdir(os.path.join(path_save, path_dataset, name_folder, 'cash_image'))
    # except:
    #     pass
    try:
        os.mkdir(os.path.join(path_save,path_dataset,name_folder,'ImageSets','Main'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path_save,path_dataset,name_folder,'Annotations'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path_save,path_dataset,name_folder,'JPEGImages'))
    except:
        pass

def create_txt_file(folder):
    all_image = set()
    type_txt_file=[]
    if folder == 'VOC2012':
        type_txt_file.append('train')
        type_txt_file.append('trainval')
        type_txt_file.append('val')
    else:
        type_txt_file.append('train')
        type_txt_file.append('trainval')
        type_txt_file.append('val')
        type_txt_file.append('test')
    for type_file in type_txt_file:
        write_image = set()
        for name_classes in classes:
            file_open=open(os.path.join(path_save,path_dataset,folder,'ImageSets','Main',f'{type_file}.txt'),'a')
            for file in os.listdir(os.path.join(path_folder,folder,'ImageSets','Main')):
                if name_classes+'_'+type_file+'.txt'==file:
                    with open(os.path.join(path_folder,folder,'ImageSets','Main',file),'r') as f:
                        for str in f:
                            str_=str.split(' ')
                            if (str_[1]!='-1\n' or len(str_)==3) and not(str_[0] in write_image):
                                file_open.write(f'{str_[0]}\n')
                                write_image.add(str_[0])
                                all_image.add(str_[0])

        print(f'{len(write_image)} type file {type_file}')
    print(f"all image {len(all_image)}")
    all_image.clear()


def main(folder):
    create_folder(folder)
    image_set=set()
    path_folder_annotation=os.path.join(path_folder,folder,'Annotations')
    for file_xml in os.listdir(path_folder_annotation):
        name_image=parseXML(os.path.join(path_folder_annotation,file_xml),folder)
        if name_image!='None':
            image_set.add(name_image)
    print(f'Process copy xml file {folder} finished')
    path_folder_image=os.path.join(path_folder,folder,'JPEGImages')
    for elem in os.listdir(path_folder_image):
        if elem in image_set:
            original_image = Image.open(os.path.join(path_folder,folder,'JPEGImages', elem))
            resized_image = original_image.resize(size)
            resized_image.save(os.path.join(path_save,path_dataset,folder,'JPEGImages', elem))
            # shutil.copyfile(os.path.join(path_folder,folder,'JPEGImages',elem),os.path.join(path_save,path_dataset,folder,'cash_image',elem))

    # resize_image(os.path.join(path_save,path_dataset,folder,'cash_image'),os.path.join(path_save,path_dataset,folder,'JPEGImages'))

    print(f'Process copy image file {folder} finished')
    create_txt_file(folder)
    print(f'Process create txt file {folder} finished')
    # os.rmdir(os.path.join(path_save, path_dataset, folder, 'cash_image'))

if __name__=="__main__":
    for folder in os.listdir(path_folder):
        main(folder)
