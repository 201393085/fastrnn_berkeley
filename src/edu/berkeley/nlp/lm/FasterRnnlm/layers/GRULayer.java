package edu.berkeley.nlp.lm.FasterRnnlm.layers;

import edu.berkeley.nlp.lm.FasterRnnlm.Matrix;
import edu.berkeley.nlp.lm.FasterRnnlm.Vector;
import edu.berkeley.nlp.lm.FasterRnnlm.layers.Activation.Activation;
import edu.berkeley.nlp.lm.FasterRnnlm.layers.Activation.SigmoidActivation;
import edu.berkeley.nlp.lm.FasterRnnlm.layers.Activation.TanhActivation;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.logging.Logger;

/**
 * Created by nlp on 16-12-6.
 */
public class GRULayer extends AbstractLayer {
    private static String strClassName = GRULayer.class.getName();
    private static Logger logger = Logger.getLogger(strClassName);
    class Weights {
        Matrix syn_reset_in_;
        Matrix syn_reset_out_;
        Matrix syn_update_in_;
        Matrix syn_update_out_;
        Matrix syn_quasihidden_in_;
        Matrix syn_quasihidden_out_;

        Vector bias_reset_;
        Vector bias_update_;
        public Weights(){
            syn_reset_in_ = new Matrix(layer_size_,layer_size_);
            syn_reset_out_ = new Matrix(layer_size_,layer_size_);
            syn_update_in_ = new Matrix(layer_size_,layer_size_);
            syn_update_out_ = new Matrix(layer_size_,layer_size_);
            syn_quasihidden_in_ = new Matrix(layer_size_,layer_size_);
            syn_quasihidden_out_ = new Matrix(layer_size_,layer_size_);

            bias_reset_ = new Vector(layer_size_);
            bias_update_ = new Vector(layer_size_);
        }
        public void Load(DataInputStream dis) {
            try{
                syn_reset_in_.ReadMatrix(dis);
                syn_reset_out_.ReadMatrix(dis);
                syn_update_in_.ReadMatrix(dis);
                syn_update_out_.ReadMatrix(dis);
                syn_quasihidden_in_.ReadMatrix(dis);
                syn_quasihidden_out_.ReadMatrix(dis);

                bias_reset_.ReadVector(dis);
                bias_update_.ReadVector(dis);
            } catch (IOException e){
                logger.severe("无法读取simplelayer");
            }
            matrices_.add(syn_reset_in_);
            matrices_.add(syn_reset_out_);
            matrices_.add(syn_update_in_);
            matrices_.add(syn_update_out_);
            matrices_.add(syn_quasihidden_in_);
            matrices_.add(syn_quasihidden_out_);

            vectors_.add(bias_reset_);
            vectors_.add(bias_update_);
        }
    }
    class GRUUpdater extends Updater{
        Matrix reset_, reset_g_;
        Matrix update_, update_g_;
        Matrix partialhidden_, partialhidden_g_;
        Matrix quasihidden_, quasihidden_g_;
        public GRUUpdater(){
            super(layer_size_);
        }

        @Override
        public void GetOutputMatrix(Matrix matrix){
            reset_ = Matrix.Multiplication(matrix, GRULayer.this.weights_.syn_reset_in_, false, true);
            update_ = Matrix.Multiplication(matrix, GRULayer.this.weights_.syn_update_in_, false, true);
            quasihidden_ = Matrix.Multiplication(matrix, GRULayer.this.weights_.syn_quasihidden_in_,false, true);
            partialhidden_ = new Matrix(matrix.row,size_);

            Activation sigmoidActivation = new SigmoidActivation();
            Activation tanhActivation = new TanhActivation();
            for(int step=0; step<matrix.row; ++step){
                if (GRULayer.this.use_bias_) {
                    for(int i=0; i<size_; ++i){
                        reset_.data[step][i] += GRULayer.this.weights_.bias_reset_.data[i];
                        update_.data[step][i] += GRULayer.this.weights_.bias_update_.data[i];
                    }
                }

                if (step != 0) {
                    for(int i=0; i<size_; ++i){
                        for(int j=0; j<size_; ++j){
                            reset_.data[step][i] += matrix.data[step-1][j] * GRULayer.this.weights_.syn_reset_out_.data[i][j];
                            update_.data[step][i] += matrix.data[step-1][j] * GRULayer.this.weights_.syn_update_out_.data[i][j];
                        }
                    }
                }
                sigmoidActivation.Forward(reset_.data[step], size_);
                sigmoidActivation.Forward(update_.data[step], size_);

                if (step != 0) {
                    for(int i=0; i<size_; ++i){
                        partialhidden_.data[step][i] = matrix.data[step-1][i] * reset_.data[step][i];
                    }
                    for(int i=0; i<size_; ++i){
                        for(int j=0; j<size_; ++j){
                            quasihidden_.data[step][i] += partialhidden_.data[step][j] * GRULayer.this.weights_.syn_quasihidden_out_.data[i][j];
                        }
                    }


                }
                tanhActivation.Forward(quasihidden_.data[step], size_);
                if (step == 0) {
                    for(int i=0; i<size_; ++i)
                        matrix.data[step][i] = quasihidden_.data[step][i] * update_.data[step][i];
                } else {
                    // output_t = (quasihidden_t - output_{t - 1}) * update_t + output_{t - 1}
                    for(int i=0; i<size_; ++i)
                        matrix.data[step][i] = (quasihidden_.data[step][i] - matrix.data[step-1][i]) * update_.data[step][i] + matrix.data[step-1][i];
                }
            }
        }


    }

    Weights weights_;
    int layer_size_;
    boolean use_bias_;
    boolean use_input_weights_;

    public GRULayer(int layer_size, boolean use_bias, boolean use_input_weights){
        this.layer_size_ = layer_size;
        this.use_bias_ = use_bias;
        this.use_input_weights_ = use_input_weights;
        this.weights_ = new Weights();
        this.updater = new GRUUpdater();
    }

    @Override
    public void Load(DataInputStream dis){
        weights_.Load(dis);
    }

}
