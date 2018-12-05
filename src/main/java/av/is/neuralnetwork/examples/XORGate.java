package av.is.neuralnetwork.examples;

import av.is.neuralnetwork.Layer;
import av.is.neuralnetwork.Network;

import java.util.Arrays;

public class XORGate {
    
    public static void main(String args[]) {
        Network network = Network.builder().learningDate(0.1d).iterations(100000)
                .inputNeurons(2)                                                           // Set up input neurons
                .addLayer(Layer.builder().neurons(4).function(Layer.FunctionType.SIGMOID)) // Hidden layer #1
                .addLayer(Layer.builder().neurons(3).function(Layer.FunctionType.SIGMOID)) // Hidden layer #2
                .addLayer(Layer.builder().neurons(1).function(Layer.FunctionType.SIGMOID)) // Last one is always output neurons
                .build();
        
        // inputNeurons(n) 의 n과 inputs[0 ~ inputs.length - 1].length는 같아야합니다.
        // 또한 마지막으로 등록된 레이어의 neurons(n) 의 n은 ouputs[0 ~ ouputs.length - 1].length와 같아야해요.
        
        double[][] inputs = new double[][] {
                { 0, 0 },
                { 0, 1 },
                { 1, 0 },
                { 1, 1 }
        };
        
        double[][] outputs = new double[][] {
                { 0 },
                { 1 },
                { 1 },
                { 0 }
        };
        
        network.train(inputs, outputs);
        analyze(new double[][] {{ 1, 1 }}, network);
        analyze(new double[][] {{ 1, 0 }}, network);
        analyze(new double[][] {{ 0, 1 }}, network);
        analyze(new double[][] {{ 0, 0 }}, network);
    }
    
    private static void analyze(double[][] test, Network network) {
        network.think(test);
        
        double[] array = network.getOutput().getMatrix()[0];
        int[] result = new int[array.length];
        for(int i = 0; i < array.length; i++) {
            double value = array[i];
            if(value >= 0.5) {
                result[i] = 1;
            } else {
                result[i] = 0;
            }
        }
        
        System.out.println(Arrays.toString(result));
    }
    
}
