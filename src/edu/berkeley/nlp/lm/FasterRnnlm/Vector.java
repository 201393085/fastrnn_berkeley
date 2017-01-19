package edu.berkeley.nlp.lm.FasterRnnlm;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * Created by nlp on 16-12-6.
 */
public class Vector {
    public double data[];
    public int length;
    public Vector(double[] v){
        this.length = v.length;
        this.data = Arrays.copyOf(v,this.length);
    }
    public Vector(int length){
        this.length = length;
        this.data = new double[length];
    }
    public void ReadVector(DataInputStream dis) throws IOException {
        for(int i=0; i<length; ++i) {
            if (Constant.USE_DOUBLE)
                data[i] = BinaryTransformer.Double(dis.readDouble());
            else
                data[i] = BinaryTransformer.Float(dis.readFloat());
        }
    }
}
