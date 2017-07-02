import java.util.Random;

public class MLP {

	private int NI; 		// Number of inputs
	private int NH; 		// Number of hidden units
	private int NO; 		// Number of outputs
	private double[][] W1;	// Weights in the lower layer
	private double[][] W2;	// Weights in the upper layer
	private double[][] dW1;	// Weights changes to W1
	private double[][] dW2;	// Weights changes to W2
	private double[] Z1;	// Activations for lower layer
	private double[] Z2;	// Activations for upper layer
	private double[] H;		// Hidden neuron values
	private double[] O;		// Outputs values
	private double[] I;		// Current Input values
	
	/**
	 * Multi-Layer Perceptron Construction
	 * 
	 * @param inputNum
	 * @param hiddenUnitNum
	 * @param outputNum
	 */
	public MLP(int inputNum, int hiddenUnitNum, int outputNum) {
		NI = inputNum;
		NH = hiddenUnitNum;
		NO = outputNum;
		
		W1 = new double[NI][NH];
		W2 = new double[NH][NO];
		H = new double[NH];
		O = new double[NO];
		Z1 = new double[NH];
		Z2 = new double[NO];
		dW1 = new double[NI][NH];
		dW2 = new double[NH][NO];
		
		randomise();
	}

	private void randomise() {
		Random r = new Random();
		
		//W1 and W2 initialised with weights in the range (-0.5, 0.5)
		for(int input = 0; input < NI; input++) {
			for(int hidden = 0; hidden < NH; hidden++) {
				W1[input][hidden] = r.nextDouble() - 0.5;
				dW1[input][hidden] = 0;
			}
		}
		for(int hidden = 0; hidden < NH; hidden++) {
			for(int output = 0; output < NO; output++) {
				W2[hidden][output] = r.nextDouble() - 0.5;
				dW2[hidden][output] = 0;
			}
		}
	}
	
	// Returns generated output
	public double[] forward(double[] Input) {		
		I = Input;
		//Lower Layer
		for(int hidden = 0; hidden < NH; hidden++) {
			Z1[hidden] = 0.0;
			for(int input = 0; input < NI; input++) {
				Z1[hidden] += Input[input] * W1[input][hidden];
			}
			//Activation function
			H[hidden] = sigmoid(Z1[hidden]);
		}
		
		//Upper Layer
		for(int output = 0; output < NO; output++) {
			Z2[output] = 0.0;
			for(int hidden = 0; hidden < NH; hidden++) {
				Z2[output] += H[hidden] * W2[hidden][output];
			}
			//Activation function
			O[output] = sigmoid(Z2[output]);
		}
		
		return O;
	}
	
	//Returns the mean squared error of the output
	public double backwards(double[] t) {
		double totalError = 0;
		double oDelta[] = new double[NO];
		
		//upper layer
		for(int output = 0; output < NO; output++) {
			oDelta[output] = t[output] - O[output];
			totalError += Math.pow(oDelta[output], 2);
			oDelta[output] *= sigmoidDeriv(O[output]);
			
			for(int hidden = 0; hidden < NH; hidden++) {
				dW2[hidden][output] += oDelta[output] * H[hidden];
			}
		}

		//lower layer
		for(int hidden = 0; hidden < NH; hidden++) {
			double hDelta = 0.0;
			for(int output = 0; output < NO; output++) {
				hDelta += (oDelta[output] * W2[hidden][output]);
			}
			hDelta *= sigmoidDeriv(H[hidden]);
			
			for(int input = 0; input < NI; input++) {
				dW1[input][hidden] += hDelta * I[input];
			}
		}
		
		return (totalError/NO);
	}
	
	public void updateWeights(double learningRate) {
		for(int input = 0; input < NI; input++) {
			for(int hidden = 0; hidden < NH; hidden++) {
				W1[input][hidden] += learningRate*dW1[input][hidden];
				dW1[input][hidden] = 0;
			}
		}
		for(int hidden = 0; hidden < NH; hidden++) {
			for(int output = 0; output < NO; output++) {
				W2[hidden][output] += learningRate*dW2[hidden][output];
				dW2[hidden][output] = 0;
			}
		}
	}
	
	//s(x) = 1/(1+e^(-x))
	private double sigmoid(double x) {
		return (1/(1+Math.exp(-x)));
	}
	
	//s'(x) = s(x)(1-s(x))
	private double sigmoidDeriv(double x) {
		return (x*(1-x)); 
	}
}
