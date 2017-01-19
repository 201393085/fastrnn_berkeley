package edu.berkeley.nlp.lm.FasterRnnlm.layers;

import edu.berkeley.nlp.lm.FasterRnnlm.Matrix;
import edu.berkeley.nlp.lm.FasterRnnlm.layers.Activation.Activation;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.logging.Logger;

/**
 * Created by nlp on 16-12-6.
 */
public class SimpleLayer extends AbstractLayer {
    private static String strClassName = SimpleLayer.class.getName();
    private static Logger logger = Logger.getLogger(strClassName);
    class Weights {
        Matrix syn_rec_;
        Matrix syn_in_;
        public Weights(){
            syn_rec_ = new Matrix(layer_size_,layer_size_);
            if(use_input_weights_)
                syn_in_ = new Matrix(layer_size_,layer_size_);
        }
        public void Load(DataInputStream dis) {
            try{
                syn_rec_.ReadMatrix(dis);
                if(use_input_weights_)
                    syn_in_.ReadMatrix(dis);
            } catch (IOException e){
                logger.severe("无法读取simplelayer");
            }
            matrices_.add(syn_rec_);
            if(use_input_weights_)
                matrices_.add(syn_in_);
        }
    }

    Weights weights_;
    int layer_size_;
    boolean use_input_weights_;
    Activation activation_;

    public SimpleLayer(int layer_size, boolean use_input_weights, Activation activation){
        this.layer_size_ = layer_size;
        this.use_input_weights_ = use_input_weights;
        this.activation_ = activation;
        this.weights_ = new Weights();
    }

    @Override
    public void Load(DataInputStream dis){
        weights_.Load(dis);
    }
}
