import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow
from tensorflow.keras.layers import Input, Activation, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.models import Model


class UNet:

    def __init__(self, height, width, channels, out_size,
                 doub_conv_layers=5,
                 padding=None,
                 pool_size=(2, 2),
                 strides=(2,2),
                 filters_layer_1=64,
                 activation='relu',
                 kernel_size=(3, 3),
                 dropout=True,
                 dropout_val=0.2,
                 upconv_kernel=(2, 2),
                 final_activation='sigmoid'):

        self.height = height
        self.width = width
        self.channels = channels
        self.doub_conv_layers = doub_conv_layers
        self.padding = padding
        self.pool_size = pool_size
        self.filters_layer_1 = filters_layer_1
        self.activation = activation
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dropout_val = dropout_val
        self.upconv_kernel = upconv_kernel
        self.out_size = out_size
        self.strides = strides
        self.final_activation = final_activation


    def doub_conv(self, input_channels, out_channels, filters):
        if self.padding == 'same':
            conv1 = Conv2D(filters=out_channels, kernel_size=filters, padding=self.padding)(input_channels)
            relu1 = Activation(self.activation)(conv1)
            conv2 = Conv2D(filters=out_channels, kernel_size=filters, padding=self.padding)(relu1)
            relu2 = Activation(self.activation)(conv2)
        else:
            conv1 = Conv2D(filters=out_channels, kernel_size=filters)(input_channels)
            relu1 = Activation(self.activation)(conv1)
            conv2 = Conv2D(filters=out_channels, kernel_size=filters)(relu1)
            relu2 = Activation(self.activation)(conv2)
        return relu2


    def crop__tensor(self, original_tensor, target_tensor):
        target_size = target_tensor.shape
        tensor_size = original_tensor.shape

        delta_h = tensor_size[1] - target_size[1]
        delta_h = delta_h//2

        delta_w = tensor_size[2] - target_size[2]
        delta_w = delta_w//2

        return original_tensor[:, delta_h:tensor_size[1] - delta_h, delta_w:tensor_size[2] - delta_w, :]



    def build_network(self):
        layer_output = []
        input = Input((self.height, self.width, self.channels), name='Input')
        c = self.doub_conv(input, self.filters_layer_1, self.kernel_size)
        layer_output.append(c)
        for i in range(self.doub_conv_layers - 1):
            p = MaxPooling2D(self.pool_size)(c)
            c = self.doub_conv(p, self.filters_layer_1 * 2 ** (i + 1), self.kernel_size)
            layer_output.append(c)
            print(c.shape)

        for i in range(self.doub_conv_layers, 1, -1):
            u = Conv2DTranspose(self.filters_layer_1 * 2 ** (i - 2), self.upconv_kernel, strides=self.strides)(c)
            if self.padding == 'same':
                try:
                    u = concatenate([u, layer_output[i - 2]])
                except:
                    u = concatenate([u, self.crop__tensor(layer_output[i - 2], u)])
            else:
                u = concatenate([u, self.crop__tensor(layer_output[i - 2], u)])

            c = self.doub_conv(u, self.filters_layer_1 * 2 ** (i -2), self.kernel_size)
            print(c.shape)

        c = Conv2D(self.out_size, (1, 1), activation=self.final_activation)(c)

        return Model(inputs=[input], outputs=[c])


unet = UNet(256, 256, 1, 2, doub_conv_layers=1, padding='same')
model = unet.build_network()
print(model.summary())
