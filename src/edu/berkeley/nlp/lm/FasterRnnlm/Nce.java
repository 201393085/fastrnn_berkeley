package edu.berkeley.nlp.lm.FasterRnnlm;

import static edu.berkeley.nlp.lm.FasterRnnlm.Constant.MAX_NGRAM_ORDER;
import static java.lang.Math.log;
import static java.lang.Math.exp;

/**
 * Created by nlp on 16-12-3.
 */
public class Nce {
    static final int kMaxNoiseSamples = 1024;
    private class NoiseSample {
        int[] noise_words;
        double[] noise_ln_probabilities;
        double target_ln_probability;
        int size;
        NoiseSample(){
            this.noise_words = new int[kMaxNoiseSamples];
            this.noise_ln_probabilities = new double[kMaxNoiseSamples];
        }
    }

    private double zln_;
    private int layer_size_;
    private int vocab_size_;
    private long maxent_hash_size_;
    private Matrix sm_embedding_;

    public Nce(
            double zln_,
            int layer_size_,
            int vocab_size_,
            long maxent_hash_size_,
            Matrix sm_embedding_
    ){
        this.zln_ = zln_;
        this.layer_size_ = layer_size_;
        this.vocab_size_ = vocab_size_;
        this.maxent_hash_size_ = maxent_hash_size_;
        this.sm_embedding_ = sm_embedding_;
    }

    private double CalculateWordLnScore(double[] hidden, MaxEnt maxent, long[] maxent_indices, int pos, int maxent_indices_count, int word){
        // Suprisingly, explicit loop works faster then the Eigen-based expression
        // Real score_rnnlm = hidden.cross(sm_embedding_.row(word));
        double score_rnnlm = 0;
        double[] embedding = word==-1?sm_embedding_.data[vocab_size_]:sm_embedding_.data[word];
        for (int i = 0; i < layer_size_; i++) {
            score_rnnlm += hidden[i] * embedding[i];
        }
        for (int i = 0; i < maxent_indices_count; i++) {
            long maxent_index = maxent_indices[i+pos]+word;
            score_rnnlm += maxent.GetValue((int)maxent_index);
        }
        return score_rnnlm - zln_;
    }

    public void CalculateLogProbabilityBatch(
        Matrix hidden_layers,
        MaxEnt maxent,
        long[] maxent_indices_all,
        int[] maxent_indices_count_all,
        int[] sentence, int sentence_length,
        boolean do_not_normalize,
        double[] logprob_per_pos
    ) {
        for (int target = 1; target <= sentence_length; ++target) {
            double[] hidden = hidden_layers.GetRow(target - 1);
            if(hidden == null) return ;

            int pos = MAX_NGRAM_ORDER * (target - 1);
            int maxent_size = maxent_indices_count_all[target - 1];
            double target_ln_score = CalculateWordLnScore(hidden, maxent, maxent_indices_all, pos, maxent_size, sentence[target]);
            if (!do_not_normalize) {
                double Z = 0;
                for (int word = 0; word < vocab_size_; ++word) {
                    Z += exp(CalculateWordLnScore(hidden, maxent, maxent_indices_all, pos, maxent_size, word));
                }
                target_ln_score -= log(Z);
            }
            // convert to ln -> log10
//            logprob_per_pos[target - 1] = target_ln_score / log(10);
            logprob_per_pos[target - 1] = target_ln_score;
        //    System.out.println("+-+-"+target_ln_score);
        }
    }
}
