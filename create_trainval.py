import os

if __name__=="__main__":
    path='D:\\Kyrs\\test3\\OIDv4_ToolKit\\open_image_to_VOC\\OIDv4_to_VOC\\res'
    vec=[]
    for file in os.listdir(path):
        file_name=os.path.splitext(file)
        print(file_name[0])