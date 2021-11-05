import tensorflow as tf
import data_handler
import weight_to_mif


print("Loading qmodel")
interpret = tf.lite.Interpreter(model_path="qmodel.tflite")
print("Allocating tensors")
interpret.allocate_tensors()

"""
Salva pesos em TXT para carregar no projeto em C
"""
data_handler.save_weights_q(interpret)

# mostra fluxo de operações do interpreter do TFLite
for opdet in interpret._get_ops_details():
    print(opdet)

"""
Salva os pesos em arquivo .mif para carregar nas memória do projeto em VHDL
"""
# CONV1
weight_to_mif.weight_to_mif(interpret.tensor(5)(), "memory_files/conv1.mif", 8, 256)
scale_entrada = interpret.get_tensor_details()[11]['quantization_parameters']['scales']
scale_pesos = interpret.get_tensor_details()[5]['quantization_parameters']['scales']
scale_saida = interpret.get_tensor_details()[12]['quantization_parameters']['scales']
scale_tensor = (scale_entrada * scale_pesos)/scale_saida
weight_to_mif.bias_scale_to_mif(interpret.tensor(8)(), scale_tensor, "memory_files/conv1_bias.mif", 32, 32)

# CONV2
weight_to_mif.weight_to_mif(interpret.tensor(4)(), "memory_files/conv2.mif", 8, 1024)
scale_entrada = interpret.get_tensor_details()[13]['quantization_parameters']['scales']
scale_pesos = interpret.get_tensor_details()[4]['quantization_parameters']['scales']
scale_saida = interpret.get_tensor_details()[14]['quantization_parameters']['scales']
scale_tensor = (scale_entrada * scale_pesos)/scale_saida
weight_to_mif.bias_scale_to_mif(interpret.tensor(9)(), scale_tensor, "memory_files/conv2_bias.mif", 32, 32)

# CONV3
weight_to_mif.weight_to_mif(interpret.tensor(3)(), "memory_files/conv3.mif", 8, 8192)
scale_entrada = interpret.get_tensor_details()[15]['quantization_parameters']['scales']
scale_pesos = interpret.get_tensor_details()[3]['quantization_parameters']['scales']
scale_saida = interpret.get_tensor_details()[16]['quantization_parameters']['scales']
scale_tensor = (scale_entrada * scale_pesos)/scale_saida
weight_to_mif.bias_scale_to_mif(interpret.tensor(6)(), scale_tensor, "memory_files/conv3_bias.mif", 32, 64)

# CONV4
weight_to_mif.weight_to_mif(interpret.tensor(2)(), "memory_files/conv4.mif", 8, 32768)
scale_entrada = interpret.get_tensor_details()[17]['quantization_parameters']['scales']
scale_pesos = interpret.get_tensor_details()[2]['quantization_parameters']['scales']
scale_saida = interpret.get_tensor_details()[18]['quantization_parameters']['scales']
scale_tensor = (scale_entrada * scale_pesos)/scale_saida
weight_to_mif.bias_scale_to_mif(interpret.tensor(7)(), scale_tensor, "memory_files/conv4_bias.mif", 32, 128)
