package proj9052507_BPN;

import java.util.Arrays;
import java.util.Random;

public class BackPropogation {

	public static class NeuralNet {

		private float[] output;
		private float[] input;
		private float[] weights;
		private float[] dweights;
		private boolean isSigmoid = true;

		public NeuralNet(int inputSize, int outputSize, Random r) {
			output = new float[outputSize];
			input = new float[inputSize + 1];
			weights = new float[(1 + inputSize) * outputSize];
			dweights = new float[weights.length];
			initWeights(r);
		}

		public int getInputSize() {
			return (input.length-1);
		}
		
		public int getOutputSize() {
			return output.length;
		}

		public void setIsSigmoid(boolean isSigmoid) {
			this.isSigmoid = isSigmoid;
		}

		public void initWeights(Random r) {
			for (int i = 0; i < weights.length; i++) {
				weights[i] = (r.nextFloat() - 0.5f) * 4f;
			}
		}

		public void setWeights(float[] weight) {
			for (int i = 0; i < weight.length; i++) {
				this.weights[i] = weight[i];
			}
		}
		
		public float[] getWeights() {
			float[] ret=new float[weights.length];
			System.arraycopy(weights, 0, ret, 0, weights.length);
			return ret;
		}

		public float[] run(float[] in) {
			System.arraycopy(in, 0, input, 0, in.length);
			input[in.length] = 1;
			int offs = 0;
			Arrays.fill(output, 0);
			for (int i = 0; i < output.length; i++) {
				for (int j = 0; j < input.length; j++) {
					output[i] += weights[offs + j] * input[j];
				}
				if (isSigmoid) {
					output[i] = (float) (1 / (1 + Math.exp(-output[i])));
				}
				offs += input.length;
			}
			float[] ret=new float[output.length];
			System.arraycopy(output, 0, ret, 0, output.length);
			return ret;
		}

		public float[] train(float[] error, float learningRate, float momentum) {
			int offs = 0;
			float[] nextError = new float[input.length];
			for (int i = 0; i < output.length; i++) {
				float d = error[i];
				if (isSigmoid) {
					d *= output[i] * (1 - output[i]);
				}
				for (int j = 0; j < input.length; j++) {
					int idx = offs + j;
					nextError[j] += weights[idx] * d;
					float dw = input[j] * d * learningRate;
					weights[idx] += /*dweights[idx] * momentum*/ + dw;
					dweights[idx] = dw;
				}
				offs += input.length;
			}
			return nextError;
		}
	}

	NeuralNet[] layers;

	public BackPropogation(int inputSize, int[] layersSize) {
		layers = new NeuralNet[layersSize.length];
		Random r = new Random(1234);
		for (int i = 0; i < layersSize.length; i++) {
			int inSize = i == 0 ? inputSize : layersSize[i - 1];
			layers[i] = new NeuralNet(inSize, layersSize[i], r);
		}
	}

	public NeuralNet getLayer(int idx) {
		return layers[idx];
	}

	public float[][] getWeights() {
		float[][] weits=new float[layers.length][];
		for (int i = 0; i < layers.length; i++) {
			weits[i]=new float[layers[i].getWeights().length];
			System.arraycopy(layers[i].getWeights(), 0, weits[i], 0, layers[i].getWeights().length);
		}
		return weits;
		
	}
	
	public float[] run(float[] input) {
		float[] actIn = new float[input.length];
		System.arraycopy(input,0,actIn,0,input.length);
		for (int i = 0; i < layers.length; i++) {
			actIn=layers[i].run(actIn);
		}
		float[] ret=new float[actIn.length];
		System.arraycopy(actIn, 0, ret, 0, actIn.length);
		return ret;
	}

	public void train(float[] input, float[] targetOutput, float learningRate, float momentum) {
		float[] calcOut = run(input);
		float[] error = new float[calcOut.length];
		for (int i = 0; i < error.length; i++) {
			error[i] = targetOutput[i] - calcOut[i]; // negative error
		}
		for (int i = layers.length - 1; i >= 0; i--) {
			error = layers[i].train(error, learningRate, momentum);
		}
	}
	
}