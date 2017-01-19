package edu.berkeley.nlp.lm.FasterRnnlm.layers.Activation;

import static java.lang.Math.max;
import static java.lang.Math.min;

/**
 * Created by nlp on 16-12-6.
 */
public class TruncatedReLUActivation implements Activation {
    private static final double kReLUTruncation = 20.0;

    @Override
    public void Forward(double[] hidden, int size) {
        for (int i = 0; i < size; i++) {
            hidden[i] = min(max(hidden[i], 0.0), kReLUTruncation);
        }
    }

    @Override
    public void Backward(double[] hidden, int size, double[] hidden_g) {
        for (int i = 0; i < size; ++i) {
            hidden_g[i] *= (hidden[i] > 0 && hidden[i] < kReLUTruncation) ? 1 : 0;
        }
    }
}
