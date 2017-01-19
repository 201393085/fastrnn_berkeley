package edu.berkeley.nlp.lm.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by nlp on 16-11-28.
 */
public class Append {
    public static void writeToTxtByFileWriter(String path, String content){
        File file = new File(path);
        BufferedWriter bw = null;
        try {
            FileWriter fw = new FileWriter(file, true);
            bw = new BufferedWriter(fw);
            bw.write(content);
        } catch (IOException e) {
            e.printStackTrace();
        }finally{
            try {
                bw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void main(String[] args) {
        for(int i=0;i<10;++i){
            writeToTxtByFileWriter("aaa.txt",new Integer(i).toString()+'\n');
        }
    }
}
