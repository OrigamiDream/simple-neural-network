package av.is.neuralnetwork;

import java.util.function.Function;

public class Matrix {
    
    private final double[][] matrix;
    
    Matrix(int x, int y) {
        this.matrix = new double[x][y];
    }
    
    Matrix(double[][] matrix) {
        this.matrix = matrix;
    }
    
    public double[][] getMatrix() {
        return matrix;
    }
    
    private int xLength() {
        return getMatrix().length;
    }
    
    private int yLength() {
        return yLength(0);
    }
    
    private int yLength(int index) {
        return getMatrix()[index].length;
    }
    
    void set(int x, int y, double value) {
        matrix[x][y] = value;
    }
    
    Matrix propagate(Matrix other) {
        if(xLength() == 0 || other.xLength() == 0 || yLength() != other.xLength()) {
            throw new IllegalArgumentException("Requires same matrix");
        }
        
        Matrix result = new Matrix(xLength(), other.yLength());
        for(int i = 0; i < xLength(); i++) {
            for(int j = 0; j < other.yLength(); j++) {
                double value = 0;
                for(int h = 0; h < yLength(); h++) {
                    value += getMatrix()[i][h] * other.getMatrix()[h][j];
                }
                result.set(i, j, value);
            }
        }
        return result;
    }
    
    Matrix subtract(Matrix other) {
        if(xLength() == 0 || other.xLength() == 0 || xLength() != other.xLength() || yLength() != other.yLength()) {
            throw new IllegalArgumentException("Requires same matrix");
        }
        Matrix result = new Matrix(xLength(), yLength());
        for(int i = 0; i < xLength(); i++) {
            for(int j = 0; j < yLength(i); j++) {
                result.set(i, j, getMatrix()[i][j] - other.getMatrix()[i][j]);
            }
        }
        return result;
    }
    
    Matrix add(Matrix other) {
        if(xLength() == 0 || other.xLength() == 0 || xLength() != other.xLength() || yLength() != other.yLength()) {
            throw new IllegalArgumentException("Requires same matrix");
        }
        Matrix result = new Matrix(xLength(), yLength());
        for(int i = 0; i < xLength(); i++) {
            for(int j = 0; j < yLength(i); j++) {
                result.set(i, j, getMatrix()[i][j] + other.getMatrix()[i][j]);
            }
        }
        return result;
    }
    
    Matrix transpose() {
        Matrix result = new Matrix(yLength(), xLength());
        for(int i = 0; i < xLength(); i++) {
            for(int j = 0; j < yLength(i); j++) {
                result.set(j, i, getMatrix()[i][j]);
            }
        }
        return result;
    }
    
    Matrix multiply(Matrix other) {
        if(xLength() != other.xLength()) {
            throw new IllegalArgumentException("Requires same matrix");
        }
        
        Matrix result = new Matrix(xLength(), yLength());
        for(int i = 0; i < xLength(); i++) {
            for(int j = 0; j < yLength(i); j++) {
                result.set(i, j, getMatrix()[i][j] * other.getMatrix()[i][j]);
            }
        }
        return result;
    }
    
    Matrix apply(Function<Double, Double> fn) {
        if(xLength() == 0 || yLength() == 0) {
            throw new IllegalArgumentException("Invalid matrix");
        }
        
        Matrix result = new Matrix(xLength(), yLength());
        for(int i = 0; i < xLength(); i++) {
            for(int j = 0; j < yLength(i); j++) {
                result.set(i, j, fn.apply(getMatrix()[i][j]));
            }
        }
        return result;
    }
}
