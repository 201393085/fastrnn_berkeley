package edu.berkeley.nlp.lm.FasterRnnlm;

import edu.berkeley.nlp.lm.StringWordIndexer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Created by nlp on 16-12-3.
 */
public class Vocabulary {
    private String OOV = "bzyunknowwordbzy";
//    private String OOV = "<unk>";
    public int kWordOOV = -1;
    private HashMap<String,Integer> hashMap = new HashMap<String,Integer>();
    private class Word{
        public Word(long freq,String word){
            this.freq = freq;
            this.word = word;
        }
        public long freq;
        public String word;
    }
    private ArrayList<Word> words_ = new ArrayList<Word>();

    private void RebuildHashMap(){
        hashMap.clear();
        for(int i=0; i<words_.size(); ++i){
            hashMap.put(words_.get(i).word,i);
        }
    }

    public int size(){
        return words_.size();
    }

    public int GetIndexByWord(String word) {
        if(hashMap.containsKey(word))
            return hashMap.get(word);
        else if(hashMap.containsKey(OOV)){
//            return -1;
            return hashMap.get(OOV);
        } else return -1;
    }

    public String GetWordByIndex(int index) {
        if (index >= 0 && index < words_.size()) {
            return words_.get(index).word;
        }
        return null;
    }

    public long GetFreqByWord(String word){
        return words_.get(GetIndexByWord(word)).freq;
    }

    public long GetFreqByIndex(int index){
        return words_.get(index).freq;
    }

    public int[] GetSentenceIndices(String sen){
        List<String> words = Arrays.asList(sen.trim().split("\\s+"));
        int[] indices = new int[words.size()+2];
        for(int i=1; i<indices.length-1; ++i){
            indices[i] = GetIndexByWord(words.get(i-1));
            if(indices[i]==-1) indices[i] = GetIndexByWord(OOV);
        }
        indices[0]=0;
        indices[indices.length-1]=0;
        return indices;
    }

    public int AddWord(Word word){
        int index = words_.size();
        words_.add(word);
        hashMap.put(word.word,index);
        return index;
    }
    public int AddWord(long freq,String word){
        return AddWord(new Word(freq,word));
    }
    public int AddWord(String word) {
        int index = words_.size();
        Word vocab_word = new Word(0,word);
        words_.add(vocab_word);
        hashMap.put(word,index);
        return index;
    }

    private class WordComparator implements Comparator<Word>{
        private boolean stable_sort;
        public WordComparator(boolean stable_sort){
            this.stable_sort = stable_sort;
        }
        public int compare(Word first, Word second){
            if(first.word.equals("</s>")){
                return 0;
            } else if(second.word.equals("</s>")){
                return 1;
            }
            long frequency_diff = first.freq - second.freq;
            if (stable_sort || frequency_diff != 0) {
                return frequency_diff < 0 ? 1 : 0;
            }
            return second.word.compareTo(first.word);
        }
    }
    public void Sort(boolean stable_sort){
        words_.sort(new WordComparator(stable_sort));
        RebuildHashMap();
    }

    public static Vocabulary Load(String fpath){
        Vocabulary vocabulary = new Vocabulary();
        try {
            File file=new File(fpath);
            if(file.isFile() && file.exists()){ //判断文件是否存在
                InputStreamReader read = new InputStreamReader(new FileInputStream(file));
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;
                while((lineTxt = bufferedReader.readLine()) != null){
                    String[] line = lineTxt.split(" ");
                    vocabulary.AddWord(Long.valueOf(line[1]),line[0]);
                }
                read.close();

                vocabulary.Sort(true);
                return vocabulary;
            }else{
                System.out.println("找不到指定的文件");
            }
        } catch (Exception e) {
            System.out.println("读取文件时内容出错");
            e.printStackTrace();
        }
        return vocabulary;
    }

    public static void main(String[] args) {
        Vocabulary vocabulary = Vocabulary.Load("model_name");
        for(int i=0;i<vocabulary.size();++i){
            System.out.print(vocabulary.GetWordByIndex(i)+" ");
            System.out.println(vocabulary.GetFreqByIndex(i));
        }
    }
}
