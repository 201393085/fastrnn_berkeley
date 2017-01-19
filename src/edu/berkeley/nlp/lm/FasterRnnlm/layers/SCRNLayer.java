package edu.berkeley.nlp.lm.FasterRnnlm.layers;

import edu.berkeley.nlp.lm.FasterRnnlm.Matrix;
import edu.berkeley.nlp.lm.FasterRnnlm.Vector;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.logging.Logger;

/**
 * Created by nlp on 16-12-6.
 */
public class SCRNLayer extends AbstractLayer {
    private static String strClassName = SCRNLayer.class.getName();
    private static Logger logger = Logger.getLogger(strClassName);
    class Weights {
        Matrix syn_context_;
        Vector decay_context_;
        Matrix syn_rec_context_;
        Matrix syn_rec_input_;
        Matrix syn_rec_hidden_;
        public Weights(){
            syn_context_ = new Matrix(context_size_,layer_size_);
            decay_context_ = new Vector(context_size_);
            syn_rec_context_ = new Matrix(layer_size_-context_size_,context_size_);
            syn_rec_input_ = new Matrix(layer_size_-context_size_,layer_size_);
            syn_rec_hidden_ = new Matrix(layer_size_-context_size_,layer_size_-context_size_);
        }
        public void Load(DataInputStream dis) {
            try{
                syn_context_.ReadMatrix(dis);
                syn_rec_context_.ReadMatrix(dis);
                syn_rec_input_.ReadMatrix(dis);
                syn_rec_hidden_.ReadMatrix(dis);

                decay_context_.ReadVector(dis);
            } catch (IOException e){
                logger.severe("无法读取simplelayer");
            }
            matrices_.add(syn_context_);
            matrices_.add(syn_rec_context_);
            matrices_.add(syn_rec_input_);
            matrices_.add(syn_rec_hidden_);

            vectors_.add(decay_context_);
        }
    }
    class SCRNUpdater extends Updater{
        Vector antidecay_context_;
        Matrix context_, context_g_;
        Matrix context_input_, context_input_g_;
        Matrix hidden_, hidden_g_;

        public SCRNUpdater(){
            super(layer_size_);
        }
        @Override
        public void GetOutputMatrix(Matrix matrix){
            if (context_size_ > 0) {
                context_input_ = Matrix.Multiplication(matrix, weights_.syn_context_, false, true);

                antidecay_context_ = new Vector(context_size_);
                for(int i=0; i<context_size_; ++i)
                    antidecay_context_.data[i] = 1.0 - weights_.decay_context_.data[i];
                context_ = new Matrix(matrix);
                for(int i=0; i<matrix.row; ++i){
                    for(int j=0; j<context_size_; ++j){
                        context_.data[i][j] *= antidecay_context_.data[j];
                    }
                }
            }
            hidden_ = Matrix.Multiplication(matrix, weights_.syn_rec_input_, false, true);

//            for (int step = 0; step < matrix.row; ++step) {
//                if (step != 0) {
//                    Matrix m = Matrix.Multiplication(new Matrix(new Vector(hidden_.data[step-1])),weights_.syn_rec_hidden_,false,true);
//                    for(int i=0; i<layer_size_-context_size_; ++i){
//                        hidden_.data[step][i] += m.data[0][i];
//                    }
//                }
//                if (context_size_ > 0) {
//                    if (step != 0) {
//                        context_.row(step).array() += decay_context_.W().array() * context_.row(step - 1).array();
//                    }
//                    hidden_.row(step).noalias() +=
//                            context_.row(step) * syn_rec_context_.W().transpose();
//                }
//                SigmoidActivation().Forward(hidden_.row(step).data(), hidden_.cols());
//            }
//
//            if (context_size_ > 0) {
//                output_.block(start, 0, steps, context_size_) = context_.middleRows(start, steps);
//            }
//            output_.block(start, context_size_, steps, hidden_size_) =
//                    hidden_.middleRows(start, steps);
        }
    }

    Weights weights_;
    int layer_size_;
    int context_size_;
    boolean use_input_weights_;

    public SCRNLayer(int layer_size, int context_size, boolean use_input_weights){
        this.layer_size_ = layer_size;
        this.context_size_ = context_size;
        this.use_input_weights_ = use_input_weights;
        this.weights_ = new Weights();
    }

    @Override
    public void Load(DataInputStream dis){
        weights_.Load(dis);
    }
}
