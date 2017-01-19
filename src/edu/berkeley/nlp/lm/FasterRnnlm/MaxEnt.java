package edu.berkeley.nlp.lm.FasterRnnlm;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static edu.berkeley.nlp.lm.FasterRnnlm.Constant.MAX_NGRAM_ORDER;
import static edu.berkeley.nlp.lm.FasterRnnlm.MaxEnt.LearningMethod.*;


/**
 * Created by nlp on 16-12-7.
 */
public class MaxEnt {
    static final long PRIMES[] = {
            108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803,
            251851741, 264197411, 330864029, 399999781, 407407183, 459258997, 479012069, 545678687,
            560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871,
            753085943, 755555077, 782715551, 790122953, 812345159, 814814293, 893826581, 923456189,
            940740127, 953085797, 985184539, 990122807};
    static final int PRIMES_SIZE = 36;

    enum LearningMethod{
        kSGD(1), kAdaGrad(2), kFTRL(3);
        int num;
        LearningMethod(int num){
            this.num = num;
        }
        int getNum(){return num;}
    }

    private final LearningMethod learningMethod = LearningMethod.kSGD;
    private final int kStride = learningMethod.getNum();

    public long hash_size_;
    public static class BigArray{
        private int MAX_ = -7 + Integer.MAX_VALUE / 1024;
        private double [][] data;
        private long length;
        public BigArray(long length){
            this.length = length;
            int num = (int)(length/MAX_) + 1;
            data = new double[num][MAX_];
        }
        public double get(long index){
            return data[(int)(index/MAX_)][(int)(index%MAX_)];
        }
        public void set(long index, double value){
            data[(int)(index/MAX_)][(int)(index%MAX_)] = value;
        }

    }
    public BigArray storage_;

    public MaxEnt(long hash_size){
        this.hash_size_ = hash_size;
    }

    public void Load(DataInputStream dis) throws IOException {
        if (hash_size_ > 0) {
            storage_ = new BigArray(hash_size_*kStride);
            if (learningMethod == kAdaGrad) {
                // when AdaGrad model is read, reset gradient sum to one
                for(long i=0; i<storage_.length; ++i) {
                    if (Constant.USE_DOUBLE)
                        storage_.set(i,BinaryTransformer.Double(dis.readDouble()));
                    else
                        storage_.set(i,BinaryTransformer.Float(dis.readFloat()));
                }
                for (long i = hash_size_; i-->0; ) {
                    storage_.set(i * kStride,storage_.get(i));
                    storage_.set(i * kStride + 1,1);
                }
            } else {
                for(long i=0; i<storage_.length; ++i) {
                    if (Constant.USE_DOUBLE)
                        storage_.set(i,BinaryTransformer.Double(dis.readDouble()));
                    else
                        storage_.set(i,BinaryTransformer.Float(dis.readFloat()));
                }
            }
        }
    }

    public double GetValue(long feature_index) {
        System.out.println("aa");
        return storage_.get(feature_index * kStride);
    }

    public static int CalculateMaxentHashIndices(int[] sen, int word_index, int maxent_order, long max_hash, boolean add_padding, long[] ngram_hashes, int pos){
        int maxent_present = (maxent_order > word_index + 1 && !add_padding) ? word_index + 1 : maxent_order;
        if (maxent_present!=0) {
            // (order < maxent_present) <--> (order < maxent_order && order <= word_index)
            for (int order = 0; order < maxent_present; ++order) {
                ngram_hashes[order+pos] = PRIMES[0] * PRIMES[1];
                for (int i = 1; i <= order; ++i) {
                    long word = (word_index - i >= 0) ? sen[word_index - i] : -1;
                    ngram_hashes[order+pos] += PRIMES[(int)((order+pos) * PRIMES[i] % PRIMES_SIZE + i) % PRIMES_SIZE] * (word + 1);

                }
                ngram_hashes[order+pos] = ngram_hashes[order+pos] % max_hash;
            }
        }
        return maxent_present;
    }
}
