package av.is.neuralnetwork.examples;

import av.is.neuralnetwork.Layer;
import av.is.neuralnetwork.Network;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.CopyOnWriteArrayList;

public class Calculator {

    private static final Object LOCK = new Object();

    private static final List<double[]> INPUTS = new CopyOnWriteArrayList<>();
    private static final List<double[]> OUTPUTS = new CopyOnWriteArrayList<>();

    public static void main(String args[]) {
        Network network = Network.builder().learningDate(0.001d).iterations(10000)
                .inputNeurons(3) // a , b , operator
                .addLayer(Layer.builder().neurons(10).function(Layer.FunctionType.TANH))
                .addLayer(Layer.builder().neurons(10).function(Layer.FunctionType.TANH))
                .addLayer(Layer.builder().neurons(10).function(Layer.FunctionType.TANH))
                .addLayer(Layer.builder().neurons(10).function(Layer.FunctionType.TANH))
                .addLayer(Layer.builder().neurons(10).function(Layer.FunctionType.TANH))
                .addLayer(Layer.builder().neurons(1).function(Layer.FunctionType.TANH)) // Output
                .build();

        new Thread(() -> {
            while(true) {
                try {
                    Thread.sleep(10);
                    if(INPUTS.size() == 0 || INPUTS.size() != OUTPUTS.size()) {
                        continue;
                    }
                    network.train(INPUTS.toArray(new double[0][]), OUTPUTS.toArray(new double[0][]));
                } catch (Exception e) {
                    System.out.println(e.getMessage());
                }
            }
        }).start();

        Scanner scanner = new Scanner(System.in);
        while(true) {
            try {
                String line = scanner.nextLine();
                line = line.replace(" ", "");
                if(line.isEmpty()) {
                    continue;
                }

                double operator;
                String[] exp;
                if(line.contains("+")) {
                    operator = 0;
                    exp = line.split("\\+");
                } else if(line.contains("-")) {
                    operator = 1;
                    exp = line.split("-");
                } else {
                    throw new IllegalArgumentException("Unknown operator");
                }
                String[] exp2 = exp[1].split("=");
                double a = Double.parseDouble(exp[0].trim());
                double b = Double.parseDouble(exp2[0].trim());
                double predict = Double.parseDouble(exp2[1].trim());
                analyze(new double[][] {{ a, b, operator }}, network);

                INPUTS.add(new double[] { a, b, operator });
                OUTPUTS.add(new double[] { predict });

                System.out.println("Inputs: " + INPUTS.size() + ", Outputs: " + OUTPUTS.size());
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }
    }

    private static void analyze(double[][] test, Network network) {
        network.think(test);

        double[] array = network.getOutput().getMatrix()[0];
        double[] result = new double[array.length];
        for(int i = 0; i < array.length; i++) {
            double value = array[i];
            result[i] = value;
        }

        System.out.println(Arrays.toString(result));
    }

}
