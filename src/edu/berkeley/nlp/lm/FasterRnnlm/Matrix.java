package edu.berkeley.nlp.lm.FasterRnnlm;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * Created by nlp on 16-12-6.
 */
public class Matrix {
    public double data[][];
    public int row;
    public int col;
    public Matrix(Matrix m){
        this.row = m.row;
        this.col = m.col;
        this.data = new double[this.row][this.col];
        for(int i=0;i<row;++i){
            for(int j=0;j<col;++j){
                this.data[i][j] = m.data[i][j];
            }
        }
    }
    public Matrix(Vector v){
        this.row = 1;
        this.col = v.length;
        this.data = new double[this.row][this.col];
        for(int i=0; i<col ;++i){
            this.data[0][i] = v.data[i];
        }
    }
    public Matrix(int row,int col){
        this.row = row;
        this.col = col;
        this.data = new double[this.row][this.col];
    }
    public void ReadMatrix(DataInputStream dis) throws IOException{
        for(int i=0; i<row; ++i){
            for(int j=0; j<col; ++j){
                if(Constant.USE_DOUBLE)
                    data[i][j] = BinaryTransformer.Double(dis.readDouble());
                else
                    data[i][j] = BinaryTransformer.Float(dis.readFloat());
            }
        }
    }
    public void Set0(){
        for(int i=0;i<row;++i) for(int j=0;j<col;++j) data[i][j]=0.0;
    }
    public double[] GetRow(int i){
        if(i>=0 && i<row) {
            double[] r = new double[col];
            for(int j=0; j<col; ++j){
                r[j]=data[i][j];
            }
            return r;
        }
        else return null;
    }
    public void print(){
        for(int i=0;i<row;++i){
            for(int j=0;j<col;++j){
                System.out.print(data[i][j]);
                System.out.print(" ");
            }
            System.out.println();
        }
    }

    public static Matrix Multiplication(Matrix m1, Matrix m2){
        return Multiplication(m1,m2,false,false);
    }
    public static Matrix Multiplication(Matrix m1, Matrix m2, boolean t1, boolean t2){
        int ROW,COL,K;
        if(t1){
            ROW = m1.col;
            K = m1.row;
        } else {
            ROW = m1.row;
            K = m1.col;
        }
        if(t2){
            COL = m2.row;
            if(K != m2.col) return null;
        } else {
            COL = m2.col;
            if(K != m2.row) return null;
        }
        Matrix matrix = new Matrix(ROW,COL);
        matrix.Set0();

        for(int i=0;i<ROW;++i) for(int j=0;j<COL;++j) for(int k=0;k<K;++k){
            if(t1 && t2){
                matrix.data[i][j] += m1.data[k][i] * m2.data[j][k];
            } else if(!t1 && t2){
                matrix.data[i][j] += m1.data[i][k] * m2.data[j][k];
            } else if(t1 && !t2){
                matrix.data[i][j] += m1.data[k][i] * m2.data[k][j];
            } else {
                matrix.data[i][j] += m1.data[i][k] * m2.data[k][j];
            }
        }
        return matrix;
    }
}
