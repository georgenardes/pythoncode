"""
Arquivo para testar projeto rebuffer
"""

import cv2 as cv
import numpy as np


def rebuffer (img, num_buffer_lines=3, with_pad=True):
    print("rebuffer")
    print(img.shape)
    imw = img.shape[1]  # largura
    imh = img.shape[0]  # altura
    if with_pad:
        imw += 2
        imh += 2

    line_buf0 = []
    line_buf1 = []
    line_buf2 = []
    line_buf_sel = 0

    linha = 0
    while linha < imh:
        coluna = 0
        while coluna < imw :
            if (linha == 0 or coluna == 0 or linha == (imh-1) or coluna == (imw-1)) and with_pad: # padd
                dado = 0
            else:
                if with_pad:
                    dado = img[linha-1, coluna-1, 0] + 128
                else:
                    dado = img[linha, coluna, 0] + 128
            if line_buf_sel == 0:
                line_buf0.append(dado)
            elif line_buf_sel == 1:
                line_buf1.append(dado)
            elif line_buf_sel == 2:
                line_buf2.append(dado)

            coluna += 1

        line_buf_sel += 1
        if line_buf_sel == num_buffer_lines:
            line_buf_sel = 0

        linha += 1

    print(img[..., 0])
    print(line_buf0)
    print(line_buf1)
    print(line_buf2)
