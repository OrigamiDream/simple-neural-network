package av.is.neuralnetwork.examples;

import av.is.neuralnetwork.Layer;
import av.is.neuralnetwork.Network;

public class ImageConvolutional {

    public static void main(String args[]) {
        Network network = Network.builder().learningDate(0.0001d).iterations(1000000)
                .inputNeurons(28 * 28)
                .addLayer(Layer.builder().neurons(16).function(Layer.FunctionType.SIGMOID))
                .addLayer(Layer.builder().neurons(16).function(Layer.FunctionType.SIGMOID))
                .build();

    }

}
