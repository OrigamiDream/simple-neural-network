package av.is.neuralnetwork;

import java.util.ArrayList;
import java.util.List;

public class Network {

    private Layer[] layers;
    private Matrix[] oLayers;
    private final double learningRate;
    private final int iterations;
    
    public static Builder builder() {
        return new Builder();
    }
    
    private Network(Builder builder) {
        this.learningRate = builder.learningRate;
        this.iterations = builder.iterations;
        
        int len = builder.layers.size();
        this.layers = new Layer[len];
        this.oLayers = new Matrix[len];
        Layer prev = null;
        for(int i = 0; i < len; i++) {
            Layer.Builder layer = builder.layers.get(i);
            if(i == 0) {
                layers[i] = new Layer(layer.functionType, layer.neurons, builder.input);
            } else {
                layers[i] = new Layer(layer.functionType, i == len - 1 ? layer.neurons - 1 : layer.neurons, prev.neurons);
            }
            prev = layers[i];
        }
    }
    
    public void think(double[][] input) {
        think(new Matrix(input));
    }
    
    private void think(Matrix input) {
        for(int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            
            if(i == 0) {
                oLayers[i] = input.propagate(layer.weights).apply(layer.functionType.getFunction());
            } else {
                oLayers[i] = oLayers[i - 1].propagate(layer.weights).apply(layer.functionType.getFunction());
            }
        }
    }
    
    public void train(double[][] input, double[][] output) {
        train(new Matrix(input), new Matrix(output));
    }
    
    public void train(Matrix input, double[][] output) {
        train(input, new Matrix(output));
    }
    
    public void train(double[][] input, Matrix output) {
        train(new Matrix(input), output);
    }
    
    private void train(Matrix input, Matrix output) {
        for(int iteration = 0; iteration < iterations; iteration++) {
            think(input);
            
            Matrix prev = null;
            Matrix[] deltas = new Matrix[oLayers.length];
            for(int i = oLayers.length - 1; i >= 0; i--) {
                Matrix layer = oLayers[i];
                Matrix delta;
                if(i == oLayers.length - 1) {
                    Matrix error = output.subtract(layer);
                    delta = error.multiply(layer.apply(layers[i].functionType.getDerivative()));
                } else {
                    assert prev != null;
                    Matrix error = prev.propagate(layers[i + 1].weights.transpose());
                    delta = error.multiply(layer.apply(layers[i].functionType.getDerivative()));
                }
                deltas[i] = delta;
                prev = delta;
            }

            for(int i = 0; i < deltas.length; i++) {
                Matrix adjustment;
                if(i == 0) {
                    adjustment = input.transpose().propagate(deltas[i]);
                } else {
                    adjustment = oLayers[i - 1].transpose().propagate(deltas[i]);
                }
                adjustment = adjustment.apply(x -> learningRate * x);
                layers[i].adjust(adjustment);
            }
            
            if(iteration % 5000 == 0) {
                System.out.println(" Training iteration " + iteration + " of " + iterations);
            }
        }
    }
    
    public Matrix getOutput() {
        return oLayers[oLayers.length - 1];
    }
    
    public static class Builder {
        
        List<Layer.Builder> layers = new ArrayList<>();
        double learningRate = 0.1;
        int input;
        int iterations;
        
        private Builder() {
        }
        
        public Builder addLayer(Layer.Builder builder) {
            layers.add(builder);
            return this;
        }
        
        public Builder inputNeurons(int inputNeurons) {
            this.input = inputNeurons;
            return this;
        }
        
        public Builder learningDate(double learningRate) {
            this.learningRate = learningRate;
            return this;
        }
        
        public Builder iterations(int iterations) {
            this.iterations = iterations;
            return this;
        }
        
        public Network build() {
            return new Network(this);
        }
    }
}
