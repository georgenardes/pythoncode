'''
Este programa converte os tensores do tflite interpreter em arquivos .mif
'''
import cv2 as cv
import os 
import numpy as np


def cabecalho (width=8, depth=1024):
    str_ret = ""
    str_ret += "-- begin_signature\n"
    str_ret += "-- ROM\n"
    str_ret += "-- end_signature\n"
    str_ret += "WIDTH="+str(width)+";\n"
    str_ret += "DEPTH="+str(depth)+";\n"
    str_ret += "ADDRESS_RADIX=UNS;\n"
    str_ret += "DATA_RADIX=BIN;\n\n"
    str_ret += "CONTENT BEGIN\n"
    print(str_ret)
    return str_ret
    

def rodape():
    str_ret = ""
    str_ret += "\nEND;\n"
    print(str_ret)
    return str_ret


def weight_to_mif(tensor, file_name, width=8, depth=256):
    index = 0
    with open(file_name, "w") as mif:
        mif.writelines(cabecalho(width=width, depth=depth))

        shape = tensor.shape
        max_addr = shape[0]*shape[1]*shape[2]*shape[3]

        # preenche restante com 0
        for i in range(depth-1, max_addr-1, -1):
            mif.writelines(
                str(i) +
                ": " + "00000000" +
                "; \n")
            index += 1

        # pesos
        for M in range(shape[0]-1, -1, -1):  # M FILTROS
            for C in range(shape[3]-1, -1, -1):  # C CANAIS
                for H in range(shape[1]-1, -1, -1):  # H LINHAS
                    for W in range(shape[2]-1, -1, -1):  # W COLUNAS
                        weigth = tensor[M, H, W, C] if tensor[M, H, W, C] >= 0 else tensor[M, H, W, C] + 256
                        bin_value = bin(weigth).replace("0b", "")
                        bin_value = bin_value.zfill(width)
                        mif.writelines(
                            str(depth - index - 1) +
                            ": " + str(bin_value) +
                            "; \n")
                        # print(depth - index - 1, ": ", bin_value)
                        index += 1

        mif.writelines(rodape())


def bias_scale_to_mif(bias_tensor, scale_tensor, file_name, width=32, depth=32):
    index = 0
    with open(file_name, "w") as mif:
        mif.writelines(cabecalho(width=width, depth=depth))

        shape = bias_tensor.shape

        max_addr = shape[0] * 2     # 2x por conta dos scales

        # preenche restante com 0
        for i in range(depth - 1, max_addr - 1, -1):
            mif.writelines(
                str(i) +
                ": " + "00000000000000000000000000000000" +
                "; \n")
            index += 1

        # shift bits
        shift_b = []
        # scale
        for i in range(len(scale_tensor) - 1, -1, -1):

            # M = M0 * 2^(-n) // M0 = M * 2^n
            M = scale_tensor[i]
            M0 = 0.0
            n = 0
            while 0.5 > M0 or M0 >= 1.0:
                M0 = M * float(2**n)
                n += 1

            shift_b.append(n-1)
            quantized = np.round(M0 * (2 ** 32)).astype(np.int64)  # note: overflow is not considered here
            bin_repr = np.binary_repr(quantized, width=32)

            print(M, M0, bin_repr, quantized, "bit shift: 32 +", n-1, "INDEX:", i)
            mif.writelines(
                str(depth - index - 1) +
                ": " + str(bin_repr) +
                "; \n")
            # print(depth - index - 1, ": ", bin_value)
            index += 1
        print("(M-1 downto 0)")
        for el in shift_b:
            print(str(el)+",", end="")
        print(" ")
        # bias
        for i in range(len(bias_tensor) - 1, -1, -1):
            weigth = bias_tensor[i] if bias_tensor[i] >= 0 else bias_tensor[i] + 2 ** 32
            bin_value = bin(weigth).replace("0b", "")
            bin_value = bin_value.zfill(width)
            mif.writelines(
                str(depth - index - 1) +
                ": " + str(bin_value) +
                "; \n")
            # print(depth - index - 1, ": ", bin_value)
            index += 1
        mif.writelines(rodape())
