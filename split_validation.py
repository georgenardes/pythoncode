import numpy as np
import os
import random

ROOT  = "C:\\Users\\User\\Desktop\\projeto_tcc\\cr_dataset\\new_train_dataset"


classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# lista diretorio do novo dataset de treino
folders = os.listdir(ROOT)
for folder in folders:
    i = 0
    subfolder = os.path.join(ROOT, folder)

    imgs = os.listdir(subfolder)

    # embaralha lista de imagens
    random.shuffle(imgs)

    # separa 100 imagens de cada classe
    for img_name in imgs:
        img_path = os.path.join(subfolder, img_name)

        new_img_path = subfolder.replace("new_train_dataset", "new_val_dataset")

        if not os.path.exists(new_img_path):
            os.mkdir(new_img_path)

        # move imagem para o novo diretorio
        os.rename(img_path, img_path.replace("new_train_dataset", "new_val_dataset"))

        i += 1
        if i == 100:    # interrompe loop em 100 iterações
            break
