package edu.berkeley.nlp.lm.FasterRnnlm.layers.Activation;

/**
 * Created by nlp on 16-12-6.
 */
public interface Activation {

    public void Forward(double[] hidden, int size);

    public void Backward(double[] hidden, int size, double[] hidden_g);
}