"""
Funções para carregar e salvar dados
"""

import tensorflow as tf
from tensorflow import keras


def load_data (path):
    loaded_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(32, 24),
        shuffle=True,
        seed=2,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False
    )

    return loaded_dataset


# indices manualmente selecionados da análise do resultado de get_tensor_details()
def save_weights_q(model):
    interpreter = model

    # indices = [5,8,4,9,3,6,2,7,20,10]
    # (PESOS, BIAS, ENTRADA, SAIDA)
    indices = [(5, 8, 11, 12), (4, 9, 13, 14), (3, 6, 15, 16), (2, 7, 17, 18)]  # indices fixos para CNN proposta

    with open("pesos_com_scale.txt", "w") as f:
        for i in indices:
            tensor = interpreter.tensor(i[0])()
            shape = tensor.shape

            f.write(str(shape[0] * shape[1] * shape[2] * shape[3]))
            f.write("\n" + str(shape[0]) + ";" + str(shape[1]) + ";" + str(shape[2]) + ";" + str(shape[3]))  # indexes
            f.write("\n")

            # pesos
            for M in range(shape[0]):  # M FILTROS
                for H in range(shape[1]):  # H LINHAS
                    for W in range(shape[2]):  # W COLUNAS
                        for C in range(shape[3]):  # C CANAIS
                            f.write(str(tensor[M, H, W, C]) + ";")

            # ===========================================================================

            tensor = interpreter.tensor(i[1])()
            shape = tensor.shape

            # qtd bias
            f.write("\n" + str(shape[0]))
            f.write("\n")

            # biases
            for M in range(shape[0]):  # M FILTROS
                f.write(str(tensor[M]) + ";")

            f.write("\n")

            # qtd scale
            f.write(str(shape[0]) + ";")
            f.write("\n")

            # Scale peso
            for M in range(shape[0]):  # M FILTROS
                scale = interpreter.get_tensor_details()[i[0]]['quantization_parameters']['scales'][M]
                f.write(str(scale) + ";")
            f.write("\n")

            # Scale ENTRADA
            scale = interpreter.get_tensor_details()[i[2]]['quantization_parameters']['scales'][0]
            f.write(str(scale) + ";")
            f.write("\n")

            # Zero ENTRADA
            zeros = interpreter.get_tensor_details()[i[2]]['quantization_parameters']['zero_points'][0]
            f.write(str(zeros) + ";")
            f.write("\n")

            # Scales SAIDA
            scale = interpreter.get_tensor_details()[i[3]]['quantization_parameters']['scales'][0]
            f.write(str(scale) + ";")
            f.write("\n")

            # Zeros SAIDA
            zeros = interpreter.get_tensor_details()[i[3]]['quantization_parameters']['zero_points'][0]
            f.write(str(zeros) + ";")
            f.write("\n")

        # ===========================================================================
        # Camada FC

        tensor = interpreter.tensor(20)()
        shape = tensor.shape

        f.write(str(shape[0] * shape[1]))
        f.write("\n" + str(shape[0]) + ";" + str(shape[1]))
        f.write("\n")

        for M in range(shape[0]):  # M Nucleos
            for C in range(shape[1]):  # C Num Caracteristicas
                f.write(str(tensor[M, C]) + ";")

        tensor = interpreter.tensor(10)()
        shape = tensor.shape

        f.write("\n" + str(shape[0]))
        f.write("\n")

        for M in range(shape[0]):  # M Nucleos
            f.write(str(tensor[M]) + ";")

        f.write("\n")

        # qtd scale
        f.write("1")
        f.write("\n")

        # Scale peso
        scale = interpreter.get_tensor_details()[20]['quantization_parameters']['scales'][0]
        f.write(str(scale) + ";")
        f.write("\n")

        # Scale ENTRADA
        scale = interpreter.get_tensor_details()[19]['quantization_parameters']['scales'][0]
        f.write(str(scale) + ";")
        f.write("\n")

        # Zero ENTRADA
        zeros = interpreter.get_tensor_details()[19]['quantization_parameters']['zero_points'][0]
        f.write(str(zeros) + ";")
        f.write("\n")

        # Scales SAIDA
        scale = interpreter.get_tensor_details()[21]['quantization_parameters']['scales'][0]
        f.write(str(scale) + ";")
        f.write("\n")

        # Zeros SAIDA
        zeros = interpreter.get_tensor_details()[21]['quantization_parameters']['zero_points'][0]
        f.write(str(zeros) + ";")



