package av.is.neuralnetwork.examples;

import av.is.neuralnetwork.Layer;
import av.is.neuralnetwork.Network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class SimpleCalculator {
    
    private static final double PLUS = 0;
    private static final double MINUS = 1;
    
    public static void main(String args[]) {
        Network network = Network.builder().learningDate(0.001d).iterations(100000)
                .inputNeurons(3)
                .addLayer(Layer.builder().neurons(10).function(Layer.FunctionType.TANH))
                .addLayer(Layer.builder().neurons(10).function(Layer.FunctionType.TANH))
                .addLayer(Layer.builder().neurons(10).function(Layer.FunctionType.TANH))
                .addLayer(Layer.builder().neurons(10).function(Layer.FunctionType.TANH))
                .addLayer(Layer.builder().neurons(10).function(Layer.FunctionType.TANH))
                .addLayer(Layer.builder().neurons(1).function(Layer.FunctionType.TANH))      // Output neurons
                .build();
        
        double[][] inputs = new double[][] {
                { 0.1, 0.0, PLUS }, { 0.1, 0.1, PLUS }, { 0.1, 0.2, PLUS }, { 0.1, 0.3, PLUS }, { 0.1, 0.4, PLUS }, { 0.1, 0.5, PLUS }, { 0.1, 0.6, PLUS }, { 0.1, 0.7, PLUS }, { 0.1, 0.8, PLUS }, { 0.1, 0.9, PLUS },
        
                { 0.0, 0.1, PLUS }, { 0.1, 0.1, PLUS }, { 0.2, 0.1, PLUS }, { 0.3, 0.1, PLUS }, { 0.4, 0.1, PLUS }, { 0.5, 0.1, PLUS }, { 0.6, 0.1, PLUS }, { 0.7, 0.1, PLUS }, { 0.8, 0.1, PLUS }, { 0.9, 0.1, PLUS },
        
                { 0.8, 0.2, PLUS },
                { 0.1, 0.1, PLUS },
                { 0.6, 0.3, PLUS },
                { 0.4, 0.5, PLUS },
                
                { 1.0, 0.0, MINUS }, { 1.0, 0.1, MINUS }, { 1.0, 0.2, MINUS }, { 1.0, 0.3, MINUS }, { 1.0, 0.4, MINUS }, { 1.0, 0.5, MINUS }, { 1.0, 0.6, MINUS }, { 1.0, 0.7, MINUS }, { 1.0, 0.8, MINUS }, { 1.0, 0.9, MINUS }, { 1.0, 1.0, MINUS },
                
                { 0.6, 0.2, MINUS },
                { 0.2, 0.1, MINUS },
                { 0.1, 0.1, MINUS },
                { 0.9, 0.2, MINUS },
        
                { 0.7, 0.6, MINUS },
                { 0.4, 0.4, MINUS },
                { 0.5, 0.1, MINUS },
                { 0.9, 0.1, MINUS },
        };
        
        double[][] outputs = new double[][] {
                { 0.1 }, { 0.2 }, { 0.3 }, { 0.4 }, { 0.5 }, { 0.6 }, { 0.7 }, { 0.8 }, { 0.9 }, { 1.0 },
        
                { 0.1 }, { 0.2 }, { 0.3 }, { 0.4 }, { 0.5 }, { 0.6 }, { 0.7 }, { 0.8 }, { 0.9 }, { 1.0 },
                
                { 1.0 },
                { 0.3 },
                { 0.9 },
                { 0.9 },
                
                { 1.0 }, { 0.9 }, { 0.8 }, { 0.7 }, { 0.6 }, { 0.5 }, { 0.4 }, { 0.3 }, { 0.2 }, { 0.1 }, { 0.0 },
                
                { 0.4 },
                { 0.1 },
                { 0.0 },
                { 0.7 },
                
                { 0.1 },
                { 0.0 },
                { 0.4 },
                { 0.8 }
        };
    
        List<double[]> inputList = new ArrayList<>(Arrays.asList(inputs));
        List<double[]> outputList = new ArrayList<>(Arrays.asList(outputs));
        
        network.train(inputList.toArray(new double[0][]), outputList.toArray(new double[0][]));
        analyze(new double[][] {{ 0.5, 0.2, PLUS }}, network);
        analyze(new double[][] {{ 0.8, 0.1, MINUS }}, network);
    
        Scanner scanner = new Scanner(System.in);
        while(true) {
            int state = scanner.nextInt();
            switch(state) {
                case 0: {
                    double a = scanner.nextDouble(); // 0.0 - 1.0
                    double b = scanner.nextDouble(); // 0.0 - 1.0
                    double c = scanner.nextDouble(); // 0 - PLUS, 1 - MINUS
                    double d = scanner.nextDouble(); // 0.0 - 1.0
                    
                    inputList.add(new double[] { a, b, c });
                    outputList.add(new double[] { d });
                    
                    network.train(inputList.toArray(new double[0][]), outputList.toArray(new double[0][]));
                    System.out.println("Retraining done.");
                    break;
                }
                
                case 1: {
                    String line = scanner.next();
                    double operator;
                    String[] split;
                    if(line.contains("+")) {
                        operator = PLUS;
                        split = line.split("\\+");
                    } else if(line.contains("-")) {
                        operator = MINUS;
                        split = line.split("-");
                    } else {
                        throw new IllegalArgumentException("Unknown operator");
                    }
                    double a = Double.parseDouble(split[0].trim());
                    double b = Double.parseDouble(split[1].trim());
    
                    analyze(new double[][] { { a, b, operator } }, network);
                    break;
                }
            }
        }
    }
    
    private static void analyze(double[][] test, Network network) {
        network.think(test);
        
        double[] array = network.getOutput().getMatrix()[0];
        double[] result = new double[array.length];
        for(int i = 0; i < array.length; i++) {
            double value = array[i];
            value = Math.round(value * 100d) / 100d;
            result[i] = value;
        }
        
        System.out.println(Arrays.toString(result));
    }
    
}
