package edu.berkeley.nlp.lm.FasterRnnlm.layers.Activation;

/**
 * Created by nlp on 16-12-6.
 */
public class ReLUActivation implements Activation {

    @Override
    public void Forward(double[] hidden, int size) {
        for (int i = 0; i < size; i++) {
            hidden[i] = (hidden[i] > 0) ? hidden[i] : 0;
        }
    }

    @Override
    public void Backward(double[] hidden, int size, double[] hidden_g) {
        for (int i = 0; i < size; ++i) {
            hidden_g[i] *= (hidden[i] > 0) ? 1 : 0;
        }
    }
}
