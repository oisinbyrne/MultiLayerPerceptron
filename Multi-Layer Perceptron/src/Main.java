import java.util.Random;
import java.util.Scanner;

public class Main {
	private static int maxEpochs = 10000;
	private static MLP NN;
	
	public static void main(String[] args) {
		double[][][] test1 = { { {0, 0}, {0} },
				   			   { {0, 1}, {1} },
				   			   { {1, 0}, {1} },
				   			   { {1, 1}, {0} }
				 			 };
		
		double[][][] test2;
		
		System.out.println("Would you like to run the XOR test or the sin() test?");
		Scanner scan = new Scanner(System.in);
		String input = "";
		do {
			System.out.println("Please enter x for XOR or s for sin()");
			input = scan.nextLine();
			if(input.equalsIgnoreCase("x")) {
				System.out.println("Training a Multi-Layer Perceptron on XOR function: 2 inputs, 2 hidden units and 1 output");
				train(test1, 2, 1);
				test(test1, 1);
				break;
			}
			else if(input.equalsIgnoreCase("s")) {
				test2 = populateSinVectors();
				System.out.println("Training a Multi-Layer Perceptron on sin() function: 4 inputs, 5 hidden units and 1 output");
				train(test2, 5, 1);
				test(test2, 1);
				break;
			}
			else continue;
		}while(true);
		
		scan.close();
	}
	
	private static void train(double[][][] examples, int NH, int NO) {
		int NI = examples[0][0].length;
		
		NN = new MLP(NI, NH, NO);
		
		double error;
		for(int e = 0; e < maxEpochs; e++) {
			error = 0;
			for(int p = 0; p < examples.length; p++) {
				NN.forward(examples[p][0]);
				error += NN.backwards(examples[p][1]);

				if(e % 10 == 0) NN.updateWeights(0.7);	// Update weights with learning rate
			}
			if(e%100 == 99) System.out.println("Error at epoch " + (e+1) + " is " + error);
		}
	}
	
	private static void test(double[][][] examples, int NO) {
		double[] result;
		double roundedResult;
		double roundedAnswer;
		double mse = 0;
		
		System.out.println("\nExample | Result | Answer\n-------------------------");
		for(int p = 0; p < examples.length; p++) {
			result = NN.forward(examples[p][0]);
			System.out.print("   " + (p+1) + "    |  ");
			
			for(int i=0; i < NO; i++) {
				roundedResult = (double) Math.round(result[i] * 100) / 100;
				roundedAnswer = (double) Math.round(examples[p][1][i] * 100) / 100;
				mse += (Math.pow((roundedResult - roundedAnswer), 2));
				if(i == 0) System.out.println(roundedResult + "  |  " + roundedAnswer);
				else {
					System.out.println("   |   " + roundedResult + " | " + roundedAnswer);
				}
			}
		}
		mse = (mse/examples.length)*10000; // Multiply by 100 for percentage and then again to round to 2 decimal places
		System.out.println("\n\nTotal MSE of test sample: " + ((double) Math.round(mse) / 100) + "%");
	}
	
	private static double[][][] populateSinVectors() {
		Random r = new Random();
		double[][][] sinVectors = new double[50][2][4];
		double sum;
		
		for(int i = 0; i < 50; i++) {
			sum = 0;
			//Store input values
			for(int j = 0; j < 4; j++) {
				//Random double in the range (-1, 1)
				sinVectors[i][0][j] = (2*r.nextDouble()) - 1;
				sum += sinVectors[i][0][j];
			}
			//Store output values
			sinVectors[i][1][0] = Math.sin(sum);
		}
		
		return sinVectors;
	}
}