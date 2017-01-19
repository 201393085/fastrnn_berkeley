package edu.berkeley.nlp.lm.FasterRnnlm.layers;

import edu.berkeley.nlp.lm.FasterRnnlm.Matrix;
import edu.berkeley.nlp.lm.FasterRnnlm.NNet;
import edu.berkeley.nlp.lm.FasterRnnlm.Vector;
import edu.berkeley.nlp.lm.FasterRnnlm.layers.Activation.*;

import java.io.DataInputStream;
import java.util.ArrayList;
import java.util.logging.Logger;

import static edu.berkeley.nlp.lm.FasterRnnlm.Constant.MAX_SENTENCE_WORDS;

/**
 * Created by nlp on 16-12-6.
 */
public abstract class AbstractLayer {
    private static String strClassName = AbstractLayer.class.getName();
    private static Logger logger = Logger.getLogger(strClassName);

    public abstract class Updater{
        public Updater(int layer_size){
            this.size_ = layer_size;
            this.input_ = new Matrix(MAX_SENTENCE_WORDS,size_);
            this.input_g_ = new Matrix(MAX_SENTENCE_WORDS,size_);
            this.output_ = new Matrix(MAX_SENTENCE_WORDS,size_);
            this.output_g_ = new Matrix(MAX_SENTENCE_WORDS,size_);
        }

        int size_;
        Matrix input_;
        Matrix input_g_;
        Matrix output_;
        Matrix output_g_;

        public abstract void GetOutputMatrix(Matrix matrix);
    }

    public ArrayList<Matrix> matrices_ = new ArrayList<Matrix>();
    public ArrayList<Vector> vectors_ = new ArrayList<Vector>();
    public Updater updater;

    public abstract void Load(DataInputStream dis);

    public static AbstractLayer CreateSingleLayer(String layer_type, int layer_size, boolean first_layer){
        if (layer_type.equals("sigmoid")) {
            return new SimpleLayer(layer_size, !first_layer, new SigmoidActivation());
        } else if (layer_type.equals("tanh")) {
            return new SimpleLayer(layer_size, !first_layer, new TanhActivation());
        } else if (layer_type.equals("relu")) {
            return new SimpleLayer(layer_size, !first_layer, new ReLUActivation());
        } else if (layer_type.equals("relu-trunc")) {
            return new SimpleLayer(layer_size, !first_layer, new TruncatedReLUActivation());
        } else if (layer_type.equals("gru")) {
            return new GRULayer(layer_size, false, false);
        } else if (layer_type.equals("gru-bias")) {
            return new GRULayer(layer_size, true, false);
        } else if (layer_type.equals("gru-insyn")) {
            return new GRULayer(layer_size, false, true);
        } else if (layer_type.equals("gru-full")) {
            return new GRULayer(layer_size, true, true);
        } else {
            String prefix = "scrn";
            String suffix = "fast";
            if (layer_type.startsWith(prefix)) {
                boolean fast = false;
                int offset = prefix.length();
                if (layer_type.startsWith(layer_type.substring(offset,offset+suffix.length()))) {
                    fast = true;
                    offset += suffix.length();
                }
                int context_size = 0;
                try{
                    context_size = Integer.valueOf(layer_type.substring(offset));
                } catch (Exception e){
                    return null;
                }
                if (context_size > layer_size) {
                    logger.severe("WARNING (SCRNLayer) context size must less than or equal to layer size\n");
                    context_size = layer_size;
                }
                return new SCRNLayer(layer_size, context_size, !fast || !first_layer);
            }
        }
        return null;
    }

    public static ArrayList<AbstractLayer> CreateLayer(String layer_type, int layer_size, int layer_count){
        ArrayList<AbstractLayer>layers = new ArrayList<AbstractLayer>();
        for(int i=0; i<layer_count; ++i){
            AbstractLayer layer = CreateSingleLayer(layer_type,layer_size,i==0);
            if(layer == null){
                layers.clear();
                return null;
            }
            layers.add(layer);
        }
        return layers;
    //    return new LayerStack(layer_size, layers);
    }
}
