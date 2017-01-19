package edu.berkeley.nlp.lm.FasterRnnlm;

import edu.berkeley.nlp.lm.FasterRnnlm.layers.AbstractLayer;
import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.io.LmReaders;
import edu.berkeley.nlp.lm.webserver.QueryHandler;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import static edu.berkeley.nlp.lm.FasterRnnlm.Constant.MAX_NGRAM_ORDER;
import static edu.berkeley.nlp.lm.FasterRnnlm.MaxEnt.CalculateMaxentHashIndices;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.pow;


/**
 * Created by nlp on 16-12-3.
 */
public class NNet {
    private static String strClassName = NNet.class.getName();
    private static Logger logger = Logger.getLogger(strClassName);

    private static final int kVersionStepSize = 10000;
    private static final int kCurrentVersion = 6;
    private static final int kMaxLayerTypeName = 64;
    private static final String kDefaultLayerType = "sigmoid";

    public class Config{
        int version;

        long layer_size;
        int layer_count;
        long maxent_hash_size;
        int maxent_order;

        boolean use_nce;
        double nce_lnz;

        boolean reverse_sentence;

        int hs_arity;

        String layer_type;
    }

    private Config cfg;
    private Vocabulary vocab;
    private Matrix embeddings;
    private Nce nce;
    private HSTree softmax_layer;
    private AbstractLayer rec_layer;
    private MaxEnt maxent_layer;


    public Config ReadHeader(DataInputStream dis){
        Config cfg = new Config();
        try{
            long quazi_layer_size = BinaryTransformer.Long(dis.readLong());
            cfg.version = (int)(quazi_layer_size / kVersionStepSize);
            System.out.println("version "+cfg.version);
            if (cfg.version < 0 || cfg.version > kCurrentVersion) {
                logger.severe("Bad version!");
                return cfg;
            }
            cfg.layer_size = quazi_layer_size % kVersionStepSize;
            System.out.println("layer_size: "+cfg.layer_size);

            cfg.maxent_hash_size = BinaryTransformer.Long(dis.readLong());
            cfg.maxent_order = BinaryTransformer.Int(dis.readInt());
            System.out.println("maxent_hash_size: "+cfg.maxent_hash_size);
            System.out.println("maxent_order: "+cfg.maxent_order);

            cfg.nce_lnz = 9;  // magic value for default lnz in old versions
            if (cfg.version == 0) {
                cfg.use_nce = false;
            } else if (cfg.version == 1) {
                cfg.use_nce = true;
            } else {
                cfg.use_nce = dis.readBoolean();
                if(Constant.USE_DOUBLE)
                    cfg.nce_lnz = BinaryTransformer.Double(dis.readDouble());
                else
                    cfg.nce_lnz = BinaryTransformer.Float(dis.readFloat());
            }
            System.out.println("use_nce: "+cfg.use_nce);
            System.out.println("nce_lnz: "+cfg.nce_lnz);

            cfg.reverse_sentence = false;
            if (cfg.version >= 3) {
                cfg.reverse_sentence = dis.readBoolean();
            }
            System.out.println("reverse_sentence: "+cfg.reverse_sentence);

            cfg.layer_type = kDefaultLayerType;
            if (cfg.version >= 4) {
                char[] chars = new char[kMaxLayerTypeName+1];
                for(int i=0; i < kMaxLayerTypeName; ++i){
                    chars[i] = BinaryTransformer.Char(dis.readByte());
                }

                cfg.layer_type = "";
                for(int i=0; i < kMaxLayerTypeName; ++i){
                    if(chars[i]!='\u0000'){
                        cfg.layer_type+=chars[i];
                    } else {
                        break;
                    }
                }
            }
            System.out.println("layer_type: "+cfg.layer_type);

            cfg.layer_count = 1;
            if (cfg.version >= 5) {
                cfg.layer_count = BinaryTransformer.Int(dis.readInt());
            }
            System.out.println("layer_count: "+cfg.layer_count);

            cfg.hs_arity = 2;
            if (cfg.version >= 6) {
                cfg.hs_arity = BinaryTransformer.Int(dis.readInt());
            }
            System.out.println("hs_arity: "+cfg.hs_arity);

        }catch (IOException e){
            logger.severe("NNet.ReadHeader出错!");
        }
        return cfg;
    }

    public Matrix ReadEmbeddings(DataInputStream dis){
        Matrix m = new Matrix(vocab.size(),(int)cfg.layer_size);
        Matrix matrix = new Matrix(vocab.size()+1,(int)cfg.layer_size);
        try{
            m.ReadMatrix(dis);
            for(int i=0;i<m.row;++i){
                for(int j=0;j<m.col;++j){
                    matrix.data[i][j] = m.data[i][j];
                }
            }
            for(int i=0;i<matrix.col;++i){
                matrix.data[vocab.size()][i]=0;
            }
        } catch (IOException e){
            logger.severe("读取词嵌入失败");
        }
        return matrix;
    }

    public Nce ReadNce(DataInputStream dis) {
        Matrix m = new Matrix(vocab.size(),(int)cfg.layer_size);
        Matrix matrix = new Matrix(vocab.size()+1,(int)cfg.layer_size);
        try{
            m.ReadMatrix(dis);
            for(int i=0;i<m.row;++i){
                for(int j=0;j<m.col;++j){
                    matrix.data[i][j] = m.data[i][j];
                }
            }
            for(int i=0;i<matrix.col;++i){
                matrix.data[vocab.size()][i]=0;
            }
        } catch (IOException e){
            logger.severe("读取词嵌入失败");
        }
        return new Nce(cfg.nce_lnz, (int)cfg.layer_size, vocab.size(), cfg.maxent_hash_size, matrix);
    }

    public HSTree ReadSoftmaxLayer(DataInputStream dis){
        HSTree hsTree = HSTree.CreateHuffmanTree(vocab, (int)cfg.layer_size, cfg.hs_arity);
        try{
            hsTree.Load(dis);
        } catch (IOException e){
            logger.severe("读取softmax_layer失败");
        }
        return hsTree;
    }

    public AbstractLayer ReadLayer(DataInputStream dis){
        AbstractLayer layer = AbstractLayer.CreateSingleLayer(cfg.layer_type,(int)cfg.layer_size,true);
        layer.Load(dis);
        return layer;
    }

    public MaxEnt ReadMaxEnt(DataInputStream dis){
        cfg.maxent_order = 0;
        MaxEnt maxEnt = new MaxEnt(cfg.maxent_hash_size);
        try{
            if(cfg.maxent_order>0)
            maxEnt.Load(dis);
        } catch (IOException e){
            logger.severe("读取maxent_layer失败");
        }
        return maxEnt;
    }

    public NNet(Vocabulary vocab,String fpath){
        this.vocab = vocab;
        Load(fpath);
    }

    private void Load(String fpath){
        try{
            FileInputStream fis=new FileInputStream(fpath);
            BufferedInputStream bis=new BufferedInputStream(fis);
            DataInputStream dis=new DataInputStream(bis);
            logger.info("loading header:");
            cfg = ReadHeader(dis);

            logger.info("loading embeddings:");
            embeddings = ReadEmbeddings(dis);
            if(cfg.use_nce){
                logger.info("loading nce:");
                nce = ReadNce(dis);
            }
            else {
                logger.info("loading SoftMaxLayer:");
                softmax_layer = ReadSoftmaxLayer(dis);
            }

            logger.info("loading layer:");
            rec_layer = ReadLayer(dis);

            logger.info("loading MaxEnt:");
            maxent_layer = ReadMaxEnt(dis);

            logger.info("finishing load:");
            dis.close();
            bis.close();
            fis.close();
            logger.info("load finished");
        } catch (IOException e){
            System.out.println("cannot open file");
        }
    }

    public double EvaluateLM(int[] sentence, boolean accurate_nce){
        Matrix output = new Matrix(sentence.length,(int)cfg.layer_size);
        for(int i=0;i<sentence.length;++i){
            int index = sentence[i];
            if(index == -1){
                index = sentence[i] = vocab.size();
            }
            output.data[i] = embeddings.data[index];
        }
        rec_layer.updater.GetOutputMatrix(output);
        int seq_length = sentence.length-1;
        double sen_logprob = 0.0;
        double[] logprob_per_pos = new double[seq_length];
        if (!cfg.use_nce) {
            // Hierarchical Softmax
            for (int target = 1; target <= seq_length; ++target) {
                long[] ngram_hashes = new long[MAX_NGRAM_ORDER];
                int maxent_present = CalculateMaxentHashIndices(
                        sentence, target, cfg.maxent_order, cfg.maxent_hash_size - vocab.size(), false, ngram_hashes, 0);
                double logprob = softmax_layer.CalculateLog10Probability(
                        sentence[target], ngram_hashes, maxent_present, true,
                        output.data[target-1], maxent_layer);
                sen_logprob += logprob;
//                System.out.println("logprob: "+logprob/log(10));
            }
        } else {
            // Noise Contrastive Estimation
            // We use batch logprob calculation to improve performance on GPU
            long[] ngram_hashes_all = new long[seq_length * MAX_NGRAM_ORDER];
            int[] ngram_present_all = new int[seq_length];

            for (int target = 1; target <= seq_length; ++target) {
                int pos = MAX_NGRAM_ORDER * (target - 1);
                int maxent_present = CalculateMaxentHashIndices(
                        sentence, target, cfg.maxent_order, cfg.maxent_hash_size - vocab.size(), false, ngram_hashes_all, pos);
                ngram_present_all[target - 1] = maxent_present;
            }

            nce.CalculateLogProbabilityBatch(
                    output, maxent_layer,
                    ngram_hashes_all, ngram_present_all,
                    sentence, seq_length,
                    !accurate_nce, logprob_per_pos
            );

            for (int i = 0; i < seq_length; ++i) {
                sen_logprob += logprob_per_pos[i];
            }
        }
        return sen_logprob;
    }


    //test
    private static NgramLanguageModel<String> readBinary(boolean isGoogleBinary, String vocabFile, String binaryFile) {
        NgramLanguageModel<String> lm = null;
        if (isGoogleBinary) {
            edu.berkeley.nlp.lm.util.Logger.startTrack("Reading Google Binary " + binaryFile + " with vocab " + vocabFile);
            lm = LmReaders.readGoogleLmBinary(binaryFile, vocabFile);
            edu.berkeley.nlp.lm.util.Logger.endTrack();
        } else {
            edu.berkeley.nlp.lm.util.Logger.startTrack("Reading LM Binary " + binaryFile);
            lm = LmReaders.readLmBinary(binaryFile);
            edu.berkeley.nlp.lm.util.Logger.endTrack();
        }
        return lm;
    }
    public static void main(String[] args) {
//        String vocab_path = "model_name";
//        String fpath = "model_name.nnet";
//     //   String tpath = "examples/ptb.test.txt";
//        String tpath = "valid.txt";

        String vocab_path = args[0];
        String fpath = args[1];
        String bpath = args[2];
        String Lambda = args[3];
        String tpath = args[4];

//        NgramLanguageModel lm = readBinary(false, null, "big_test.binary");
//        NgramLanguageModel lm = readBinary(false, null, "bzy_0516_google_555.model");
        NgramLanguageModel lm = readBinary(false, null, bpath);
//        NgramLanguageModel lm = null;

        Vocabulary vocab = Vocabulary.Load(vocab_path);
        NNet nnet = new NNet(vocab,fpath);
        try{
            double t1  = System.currentTimeMillis();
            BufferedReader bf= new BufferedReader(new FileReader(tpath));
            boolean kNCEAccurate = false;
            double lambda = Double.valueOf(Lambda);
            double score1 = 0;
            double score2 = 0;
            double score3 = 0;
            int cnt = 0;

            List l1 = new ArrayList<Double>();
            List l2 = new ArrayList<Double>();

            int good = 0;
            int count = 0;
            String[] sentences = {"good call I a friend cannot", "I cannot be called a good friend."};
        //    for(String sentence : sentences){
            String sentence,sentence2;  while((sentence = bf.readLine())!=null){
                bf.readLine();
                sentence2 = bf.readLine();
                bf.readLine();
            //    if(sentence.length()<5)continue;
                System.out.println("aaaaa "+sentence);
                System.out.println("bbbbb "+sentence2);
                double s1 = 0.0, s2 = 0.0;
                double ss1 = 0.0, ss2 = 0.0;
                int [] sen = vocab.GetSentenceIndices(sentence);
                int [] sen2 = vocab.GetSentenceIndices(sentence2);
                List<String> words = Arrays.asList(sentence.trim().split("\\s+"));
                List<String> words2 = Arrays.asList(sentence2.trim().split("\\s+"));
            //    if(words.size()>=30)continue;
                if (lambda >= 1.0) {
                    if(sen ==null) s1 = 0;
                    else s1 = nnet.EvaluateLM(sen, kNCEAccurate)/log(2);
                    if(sen2 ==null) ss1 = 0;
                    else ss1 = nnet.EvaluateLM(sen2, kNCEAccurate)/log(2);
                } else if(lambda <= 0.0){
                    s2 = lm.scoreSentence(words);
                    ss2 = lm.scoreSentence(words2);
                } else {
                    if(sen ==null) s1 = 0;
                    else s1 = nnet.EvaluateLM(sen, kNCEAccurate)/log(2);
                    s2 = lm.scoreSentence(words);
                    if(sen2 ==null) ss1 = 0;
                    else ss1 = nnet.EvaluateLM(sen2, kNCEAccurate)/log(2);
                    ss2 = lm.scoreSentence(words2);
                    l1.add(s1);
                    l1.add(ss1);
                    l2.add(s2);
                    l2.add(ss2);
                }
                System.out.println(s1+" "+s2+" "+((lambda)*s1+(1-lambda)*s2));

                score1+=s1;
                score2+=s2;
                score3+=(lambda)*s1+(1-lambda)*s2;

                double sss1 = (lambda)*s1+(1-lambda)*s2;
                double sss2 = (lambda)*ss1+(1-lambda)*ss2;
                count++;
                System.out.println(sss1+" "+sss2);
                good+=sss1>sss2?1:0;
                cnt+=words.size()+1;
            }
            score1 = -score1/cnt;
            score2 = -score2/cnt;
            score3 = -score3/cnt;
            double ppl1 = exp(score1*log(2));
            double ppl2 = exp(score2*log(2));
            double ppl3 = exp(score3*log(2));
            System.out.println("PPL1: "+ ppl1);
            System.out.println("PPL2: "+ ppl2);
            System.out.println("PPL3: "+ ppl3);

            double time = System.currentTimeMillis() - t1;
            System.out.println("time: "+time);
            System.out.println(good+" "+count);
            System.out.println("good rate: "+(double)good/count);

            for(int i=0;i<=100;++i){
                int ccnt=0, ggood = 0;
                double lam = i/100.0;
                for(int j=0;j<l1.size();j+=2){
                    double s1 = lam * (Double)l1.get(j) + (1-lam)*(Double)l2.get(j);
                    double s2 = lam * (Double)l1.get(j+1) + (1-lam)*(Double)l2.get(j+1);
                    if(s1>s2)ggood++;
                    ccnt++;
                }
                System.out.println("lambda: "+lam+" good rate: "+(double)ggood/ccnt);
            }
            System.out.println("time2: "+(System.currentTimeMillis() - t1 - time));
        } catch (IOException e){
            System.out.println("fail");
        }
    }
}
