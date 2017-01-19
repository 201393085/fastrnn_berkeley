package edu.berkeley.nlp.lm.FasterRnnlm.layers.Activation;

import static java.lang.Math.exp;

/**
 * Created by nlp on 16-12-6.
 */
public class SigmoidActivation implements Activation {
    @Override
    public void Forward(double[] hidden, int size) {
        for (int i = 0; i < size; i++) {
            hidden[i] = exp(hidden[i]) / (1 + exp(hidden[i]));
        }
    }

    @Override
    public void Backward(double[] hidden, int size, double[] hidden_g) {
        for (int i = 0; i < size; ++i) {
            hidden_g[i] *= hidden[i] * (1 - hidden[i]);
        }
    }
}
