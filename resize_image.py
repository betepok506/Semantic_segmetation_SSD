import os
import cv2 as cv
import numpy as np
from PIL import Image
from lxml import etree, objectify
import sys
size=(300,300)

def resize_image(path_sourse,path_dest):
    for img_file in os.listdir(path_sourse):
        original_image = Image.open(os.path.join(path_sourse,img_file))
        resized_image = original_image.resize(size)
        resized_image.save(os.path.join(path_dest,img_file))

def rewrite_xml(path,path_save):
    for file_xml in os.listdir(path):
        with open(os.path.join(path,file_xml)) as fobj:
            xml = fobj.read()

        root = etree.fromstring(xml)

        width=0
        height=0

        for appt in root.getchildren():
            if appt.tag == 'size' or appt.tag=='object':
                for elem in appt.getchildren():
                    if appt.tag=='size':
                        if elem.tag=='width':
                            width=int(elem.text)
                            elem.text=f'{size[0]}'
                        elif elem.tag=='height':
                            height=int(elem.text)
                            elem.text=f'{size[1]}'
                    if appt.tag=='object':
                        if elem.tag=='bndbox':
                            koef_y = size[0] / width
                            koef_x=size[1]/height
                            # (d - c) / (b - a) * (x - a) + c
                            for coord in elem.getchildren():
                                # coord.text = f'{int(300/((width-height)*(int(coord.text)-height)))}'
                                if coord.tag=='ymin' or coord.tag=='ymax':
                                    coord.text=f'{int(int(coord.text)*koef_x)}'
                                if coord.tag=='xmin' or coord.tag=='xmax':
                                    coord.text=f'{int(int(coord.text)*koef_y)}'


        # path_save='D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\ggt'
        obj_xml = etree.tostring(root, pretty_print=True,
                                 xml_declaration=False)
                                 # encoding='UTF-8')

        try:
            with open(f"{os.path.join(path_save,file_xml)}",
                      "wb") as xml_writer:
                xml_writer.write(obj_xml)
        except IOError:
            pass

def check():
    path_res_img='D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\\new_image'
    path_res_xml='D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\\annot'
    for file_xml in os.listdir(path_res_xml):
        with open(os.path.join(path_res_xml,file_xml)) as fobj:
            xml = fobj.read()

        root = etree.fromstring(xml)
        for img_file in os.listdir(path_res_img):
            if img_file.split('.')[0]==file_xml.split('.')[0]:
                img=cv.imread('D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\\new_image'+'\\'+file_xml.split('.')[0]+'.jpg')
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
        path_save='D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\\new_im_pr'
        cv.imwrite(f"{os.path.join(path_save,file_xml.split('.')[0]+'.jpg')}",img)

def check2():
    path_res_img='D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\image'
    path_res_xml='D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\\new_xml'
    for file_xml in os.listdir(path_res_xml):
        with open(os.path.join(path_res_xml,file_xml)) as fobj:
            xml = fobj.read()

        root = etree.fromstring(xml)
        for img_file in os.listdir(path_res_img):
            if img_file.split('.')[0]==file_xml.split('.')[0]:
                img=cv.imread('D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\image'+'\\'+file_xml.split('.')[0]+'.jpg')
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
        path_save='D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\\im_pr'
        cv.imwrite(f"{os.path.join(path_save,file_xml.split('.')[0]+'.jpg')}",img)

if __name__=="__main__":

    argv = sys.argv
    path_sourse_img = ''
    path_dest_img = ''
    path_sourse_xml=''
    path_dest_xml=''
    for a in range(len(argv)):
        if argv[a] == '--sourcepath_img':
            path_sourse_img = argv[a + 1]
        elif argv[a] == '--dest_path_img':
            path_dest_img = argv[a + 1]
        elif argv[a] == '--dest_path_xml':
            path_dest_xml = argv[a + 1]
        elif argv[a] == '--sourcepath_xml':
            path_sourse_xml = argv[a + 1]
    # check2()
    print('Start resize img')
    resize_image(path_sourse_img,path_dest_img)
    # resize_image('D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\image','D:\Kyrs\\test3\OIDv4_ToolKit\open_image_to_VOC\OIDv4_to_VOC\\new_image')
    print('Resize img finished')
    print('Start rewrite xml')
    rewrite_xml(path_sourse_xml,path_dest_xml)
    print('Finished rewrite xml')