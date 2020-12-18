import os
import sys
import shutil

def create_dir():
    try:
        os.mkdir(os.path.join(path_dest, 'OpenImage'))
    except:
        pass

    try:
        os.mkdir(os.path.join(path_dest, 'OpenImage', 'Annotations'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path_dest, 'OpenImage', 'ImageSets'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path_dest, 'OpenImage', 'ImageSets', 'Main'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path_dest, 'OpenImage', 'JPEGImages'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path_dest, 'OpenImage', 'cash'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path_dest, 'OpenImage', 'cash_xml'))
    except:
        pass

if __name__=="__main__":
    path=""
    argv=sys.argv
    path_sourse=''
    path_dest=''
    for a in range(len(argv)):
        if argv[a]=='--sourcepath':
            path_sourse=argv[a+1]
        elif argv[a]=='--dest_path':
            path_dest=argv[a+1]

    create_dir()
    for folder in os.listdir(path_sourse):
        os.system(f"python OIDv4_to_VOC.py --sourcepath {os.path.join(path_sourse,folder)} --dest_path {os.path.join(path_dest,'OpenImage','cash_xml')}")

    mas_img=[]
    for xml_file in os.listdir(os.path.join(path_dest,'OpenImage','cash_xml')):
        mas_img.append(os.path.basename(xml_file).split('.')[0])

    koef=20
    cnt_test=len(mas_img)//100 * koef
    index=0
    with open(os.path.join(path_dest,'OpenImage','ImageSets','Main','test.txt'),'w') as f:
        for img_name in range(cnt_test):
            f.write(f'{mas_img[img_name]}\n')
            index+=1

    with open(os.path.join(path_dest,'OpenImage','ImageSets','Main','train.txt'),'w') as f:
        for img_name in range(index,len(mas_img)):
            f.write(f'{mas_img[img_name]}\n')
            index+=1
    for folder in os.listdir(path_sourse):
        for img_file in os.listdir(os.path.join(path_sourse,folder)):
            if not os.path.isdir(os.path.join(path_sourse,folder,img_file)):
                shutil.copyfile(os.path.join(path_sourse,folder,img_file),
                                os.path.join(path_dest,'OpenImage','cash',img_file))

    os.system(
        f"python resize_image.py --sourcepath_img {os.path.join(path_dest, 'OpenImage','cash')} --dest_path_img {os.path.join(path_dest, 'OpenImage', 'JPEGImages' )} "
        f"--sourcepath_xml {os.path.join(path_dest,'OpenImage','cash_xml')} --dest_path_xml {os.path.join(path_dest,'OpenImage','Annotations')}")

    for file_del in os.listdir(os.path.join(path_dest,'OpenImage','cash_xml')):
        os.remove(os.path.join(path_dest,'OpenImage','cash_xml',file_del))

    for file_del in os.listdir(os.path.join(path_dest,'OpenImage','cash')):
        os.remove(os.path.join(path_dest, 'OpenImage', 'cash', file_del))

    os.rmdir(os.path.join(path_dest,'OpenImage','cash'))
    os.rmdir(os.path.join(path_dest, 'OpenImage', 'cash_xml'))
