package edu.berkeley.nlp.lm.FasterRnnlm;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * Created by nlp on 16-12-5.
 * 用于c++二进制转java二进制.
 */
public class BinaryTransformer {

    public static short Short(short s){
        short s0 = (short)((s >> 8) & 0x00ff);
        short s1 = (short)((s << 8) & 0xff00);
        return (short)(s0|s1);
    }

    public static int Int(int i){
        int i0 = (i >> 24) & 0x000000ff;
        int i1 = (i >>  8) & 0x0000ff00;
        int i2 = (i <<  8) & 0x00ff0000;
        int i3 = (i << 24) & 0xff000000;
        return i0|i1|i2|i3;
    }

    public static long Long(long l){
        long l0 = (l >> 56) & 0x00000000000000ffL;
        long l1 = (l >> 40) & 0x000000000000ff00L;
        long l2 = (l >> 24) & 0x0000000000ff0000L;
        long l3 = (l >>  8) & 0x00000000ff000000L;
        long l4 = (l <<  8) & 0x000000ff00000000L;
        long l5 = (l << 24) & 0x0000ff0000000000L;
        long l6 = (l << 40) & 0x00ff000000000000L;
        long l7 = (l << 56) & 0xff00000000000000L;
        return l0|l1|l2|l3|l4|l5|l6|l7;
    }

    public static char Char(byte b){
        return (char)(b|0x0000);
    }

    public static float Float(float f){
        int i = Float.floatToIntBits(f);
        return Float.intBitsToFloat(Int(i));
    }

    public static double Double(double d){
        long l = Double.doubleToLongBits(d);
        return Double.longBitsToDouble(BinaryTransformer.Long(l));
    }

    //测试用
    public static void main(String[] args) {
        System.out.println(Double.NaN);
    }
}
