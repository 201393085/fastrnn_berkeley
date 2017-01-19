package edu.berkeley.nlp.lm.Python;

import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.io.LmReaders;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.log;

/**
 * Created by nlp on 16-12-22.
 */
public class Main {
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

    private NgramLanguageModel lm;

    public Main(String path){
        this.lm = readBinary(false, null ,path);
    }

    public double score(String sentence){
        List<String> words = Arrays.asList(sentence.trim().split("\\s+"));
        return lm.scoreSentence(words);
    }

    public static void main(String[] args) {
        String s="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzz.,?:;'\"!@%";
        Random ra =new Random();
        Main m = new Main("char_test2.binary");
        long a=System.currentTimeMillis();

        System.out.println(System.currentTimeMillis() - a);
        System.out.println(m.score("*"));
    }
}
