package av.is.neuralnetwork;

import java.util.function.Function;

public class Layer {
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static class Builder {
        
        FunctionType functionType;
        int neurons;
        
        private Builder() {
        }
        
        public Builder function(FunctionType functionType) {
            this.functionType = functionType;
            return this;
        }
        
        public Builder neurons(int neurons) {
            this.neurons = neurons + 1;
            return this;
        }
    }
    
    public enum FunctionType {
        SIGMOID(x -> 1d / (1d + Math.exp(-x)),
                x -> x * (1 - x)),
        
        TANH(Math::tanh,
             x -> 1 - Math.tanh(x) * Math.tanh(x)),
        
        RELU(x -> (x > 0) ? x : 0d,
             x -> (x > 0) ? 1d : 0d);
        
        private final Function<Double, Double> function;
        private final Function<Double, Double> derivative;
    
        FunctionType(Function<Double, Double> function, Function<Double, Double> derivative) {
            this.function = function;
            this.derivative = derivative;
        }
    
        public Function<Double, Double> getFunction() {
            return function;
        }
    
        public Function<Double, Double> getDerivative() {
            return derivative;
        }
    }
    
    Matrix weights;
    FunctionType functionType;
    int neurons;
    
    Layer(FunctionType functionType, int neuronNum, int inputNum) {
        this.weights = new Matrix(inputNum, neuronNum);
        this.functionType = functionType;
        this.neurons = neuronNum;
        
        for(int i = 0; i < inputNum; i++) {
            for(int j = 0; j < neuronNum; j++) {
                if(i == inputNum - 1) {
                    weights.set(i, j, 1);
                } else {
                    weights.set(i, j, 0);
                }
            }
        }
    }
    
    void adjust(Matrix adjust) {
        weights = weights.add(adjust);
    }
}