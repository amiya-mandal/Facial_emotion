import PIL
from PIL import Image
import os
from tqdm import tqdm
size = 48, 48



def DirProcess(dir_name="jaffe"):
    all_files = os.listdir(dir_name)
    count = 0
    kval = None
    print(len(all_files))
    all_files = tqdm(all_files)
    for i in all_files:
        if ".tiff" in i:
            filename = dir_name+"/"+i
            data_image = Image.open(filename)
            image = data_image.resize(size)
            if "NE" in i:
                kval = 0
            elif "HA" in i:
                kval = 1
            elif "SA" in i:
                kval = 2
            elif "SU" in i:
                kval = 3
            elif "AN" in i:
                kval = 4
            elif "DI" in i:
                kval = 5
            elif "FE" in i:
                kval = 6
            name = "process/"+str(kval)+"_"+str(count)+'.tiff'
            count+=1
            image.save(name)

if __name__ == '__main__':
    DirProcess()
