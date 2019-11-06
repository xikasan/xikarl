# coding: utf-8

import tensorflow as tf
tk = tf.keras


class MLP(tk.Model):

    def __init__(
            self,
            units,
            dim_input,
            dim_output,
            scale=1,
            dtype=tf.float32,
            activation=tk.layers.ReLU,
            output_activation=None,
            name="MLP",
            before_step_func=None,
            after_step_func=None
    ):
        super().__init__(name=name)
        self.activation = activation
        self.scale = tf.constant(scale, name=name+"/scale", dtype=dtype)

        self._layers = []
        # build hidden layers
        for l, unit in enumerate(units):
            layer = tk.layers.Dense(unit, name=name+"/L{}".format(l+1), dtype=dtype)
            self._layers.append(layer)
            layer = activation(name=name+"/A{}".format(l+1))
            # layer = tk.layers.ReLU(name=name+"/A{}".format(l+1))
            self._layers.append(layer)

        # build output layer
        layer = tk.layers.Dense(dim_output, name=name+"/Lo", dtype=dtype)
        self._layers.append(layer)
        if output_activation is not None:
            layer = output_activation(name=name+"/Ao")
            self._layers.append(layer)

        # before step func
        self.before_step_func = before_step_func
        self.after_step_func = after_step_func

        # register layers to tensorflow
        self(tk.Input((dim_input, ), 1, dtype=dtype))

    def call(self, inputs):
        feature = inputs if self.before_step_func is None else self.before_step_func(inputs)
        feature = self._forward_body(feature)
        outputs = feature if self.after_step_func is None else self.after_step_func(feature)
        return outputs

    # @tf.function
    def _forward_body(self, feature):
        for layer in self._layers:
            feature = layer(feature)
        outputs = feature * self.scale
        return outputs


class Tanh(tk.Model):

    def __init__(self, name="Tanh"):
        super().__init__(name=name)
        self.func = tk.layers.Activation(tf.math.tanh)

    def call(self, inputs):
        return self.func(inputs)
