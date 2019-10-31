from keras import backend as K
from keras.layers import Conv2D, Dense

class MaskedConv2D(Conv2D):
    def flip(self):
        tmp = self.non_trainable_weights
        self.non_trainable_weights = self.trainable_weights
        self.trainable_weights = tmp
        
        
    def build(self, input_shapes, **kwargs):
        super().build(input_shapes, **kwargs)
        self.trainable_weights = []
        self.kernel_mask = self.add_weight(shape=K.int_shape(self.kernel),
                                           initializer='ones',
                                           name='kernel_mask',
                                           trainable=True)
        
        if self.use_bias:
            self.non_trainable_weights = [self.kernel, self.bias]
            
            self.bias_mask = self.add_weight(shape=K.int_shape(self.bias),
                                             initializer='ones',
                                             name='bias_mask',
                                             trainable=True)
        else:
            self.non_trainable_weights = [self.kernel]

    
    def call(self, inputs):
        def make_mask(mask_logits):
            mask_probits = K.sigmoid(mask_logits)
            mask = K.cast(K.greater(mask_probits, .5), 'float32')
            return (mask_probits + K.stop_gradient(mask - mask_probits))
        
        kernel = make_mask(self.kernel_mask) * self.kernel
        conv = K.conv2d(inputs,
                        kernel,
                        strides=self.strides,
                        padding=self.padding,
                        data_format=self.data_format,
                        dilation_rate=self.dilation_rate)
        
        if self.use_bias:
            bias = make_mask(self.bias_mask) * self.bias
            conv = K.bias_add(conv,
                              bias,
                              data_format=self.data_format)
        return conv
    
class MaskedDense(Dense):
    def flip(self):
        tmp = self.non_trainable_weights
        self.non_trainable_weights = self.trainable_weights
        self.trainable_weights = tmp
        
        
    def build(self, input_shapes, **kwargs):
        super().build(input_shapes, **kwargs)
        self.trainable_weights = []
        self.kernel_mask = self.add_weight(shape=K.int_shape(self.kernel),
                                           initializer='ones',
                                           name='kernel_mask',
                                           trainable=True)
        
        if self.use_bias:
            self.non_trainable_weights = [self.kernel, self.bias]
            
            self.bias_mask = self.add_weight(shape=K.int_shape(self.bias),
                                             initializer='ones',
                                             name='bias_mask',
                                             trainable=True)
        else:
            self.non_trainable_weights = [self.kernel]

    
    def call(self, inputs):
        def make_mask(mask_logits):
            mask_probits = K.sigmoid(mask_logits)
            mask = K.cast(K.greater(mask_probits, .5), 'float32')
            return (mask_probits + K.stop_gradient(mask - mask_probits))
        
        kernel = make_mask(self.kernel_mask) * self.kernel
        output = K.dot(inputs, kernel)
        if self.use_bias:
            bias = make_mask(self.bias_mask) * self.bias
            output = K.bias_add(output, bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output