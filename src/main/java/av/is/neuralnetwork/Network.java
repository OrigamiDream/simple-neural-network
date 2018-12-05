package av.is.neuralnetwork;

import avis.juikit.Juikit;

import javax.swing.*;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class Network {

    private List<Double> diffs = new CopyOnWriteArrayList<>();

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

        Juikit.createFrame()
                .antialiasing(true)
                .title("Simple Neural Network")
                .size(800, 1400)
                .closeOperation(WindowConstants.EXIT_ON_CLOSE)
                .repaintInterval(10)
                .painter((juikit, graphics) -> {
                    int count = juikit.width();
                    List<Double> d = new CopyOnWriteArrayList<>(diffs);

                    boolean first = true;
                    double previous = -1;
                    for(int i = d.size() - 1; i >= 0; i--) {
                        count--;
                        double value = d.get(i) * 1000;
                        graphics.fillOval(count, (int) (juikit.height() - value), 2, 2);
                        if(previous != -1) {
                            graphics.drawLine(count + 1, (int) (juikit.height() - previous), count, (int) (juikit.height() - value));
                        }
                        if(first) {
                            graphics.drawString(new BigDecimal(d.get(i)).toPlainString(), 50, 50);
                            first = false;
                        }
                        previous = value;
                        if(count == -1) {
                            break;
                        }
                    }
                })
                .visibility(true);
    }
    
    public void think(double[][] input) {
        think(new Matrix(input));
    }
    
    private void think(Matrix input) {
        for(int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            
            if(i == 0) {
                oLayers[i] = input.multiply(layer.weights).apply(layer.functionType.getFunction());
            } else {
                oLayers[i] = oLayers[i - 1].multiply(layer.weights).apply(layer.functionType.getFunction());
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
                    diffs.add(error.average());
                    delta = error.scalar(layer.apply(layers[i].functionType.getDerivative()));
                } else {
                    assert prev != null;
                    Matrix error = prev.multiply(layers[i + 1].weights.transpose());
                    delta = error.scalar(layer.apply(layers[i].functionType.getDerivative()));
                }
                deltas[i] = delta;
                prev = delta;
            }

//            double sum = 0;
            for(int i = 0; i < deltas.length; i++) {
                Matrix adjustment;
                if(i == 0) {
                    adjustment = input.transpose().multiply(deltas[i]);
                } else {
                    adjustment = oLayers[i - 1].transpose().multiply(deltas[i]);
                }
                adjustment = adjustment.apply(x -> learningRate * x);

//                sum += adjustment.average();
                layers[i].adjust(adjustment);
            }
//            diffs.add(sum / (int) deltas.length);
            
            if(iteration % 5000 == 0) {
//                System.out.println(" Training iteration " + iteration + " of " + iterations);
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
