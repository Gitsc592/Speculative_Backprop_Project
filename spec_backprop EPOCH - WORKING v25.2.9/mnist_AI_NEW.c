#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>
#include <stdbool.h>
#include <omp.h>
#include <time.h>

#define LAYER1_N_COUNT 784
#define LAYER2_N_COUNT 16
#define LAYER3_N_COUNT 16
#define LAYER4_N_COUNT 10
#define PIXELS_PER_IMAGE 784
#define TRAINING_IMAGES 60000
#define TEST_IMAGES 10000
#define LABEL_START_OFFSET 8
#define PIXEL_START_OFFSET 16
#define WEIGHT_INIT_DIVIDER 100 // To prevents weights from being too big for the RELU
#define BATCH_SIZE 15
#define LEARNING_RATE 0.01
#define MAX_GRAD 5
#define THRESHOLD 0.25
#define BETA 0
#define ALPHA 1
#define EPOCH 1

#define Train_Data_File_Pixels "mnist_data/train-images.idx3-ubyte"
#define Train_Data_File_Labels "mnist_data/train-labels.idx1-ubyte"
#define Test_Data_File_Pixels "mnist_data/t10k-images.idx3-ubyte"
#define Test_Data_File_Labels "mnist_data/t10k-labels.idx1-ubyte"

struct neuron_L1 {
	double activation;
	double prev_activation[LAYER4_N_COUNT];
	double weights[LAYER2_N_COUNT];
};

struct neuron_L2 {
	double activation;
	double sum;
	double prev_activation[LAYER4_N_COUNT];
	double prev_sum[LAYER4_N_COUNT];
	double weights[LAYER3_N_COUNT];
	double error_der;
	double bias, biasgrad, biasgrad_spec;
};

struct neuron_L3 {
	double activation;
	double sum;
	double prev_activation[LAYER4_N_COUNT];
	double prev_sum[LAYER4_N_COUNT];
	double weights[LAYER4_N_COUNT];
	double error_der;
	double bias, biasgrad, biasgrad_spec;
};

struct neuron_L4 {
	double activation;
	double sum;
	double prev_activation[LAYER4_N_COUNT];
	double prev_sum[LAYER4_N_COUNT];
	double error_der;
	double bias, biasgrad, biasgrad_spec;
};

struct neuron_acc {
	double activation;
};

// Initialize Gradients for each layer
double gradient_L1[LAYER1_N_COUNT][LAYER2_N_COUNT] = { 0 };
double gradient_L2[LAYER2_N_COUNT][LAYER3_N_COUNT] = { 0 };
double gradient_L3[LAYER3_N_COUNT][LAYER4_N_COUNT] = { 0 };
double gradient_L1_spec[LAYER1_N_COUNT][LAYER2_N_COUNT] = { 0 };
double gradient_L2_spec[LAYER2_N_COUNT][LAYER3_N_COUNT] = { 0 };
double gradient_L3_spec[LAYER3_N_COUNT][LAYER4_N_COUNT] = { 0 };
struct neuron_L1 input[LAYER1_N_COUNT];
struct neuron_L2 layer_2[LAYER2_N_COUNT];
struct neuron_L3 layer_3[LAYER3_N_COUNT];
struct neuron_L4 output[LAYER4_N_COUNT];
struct neuron_acc output_acc[LAYER4_N_COUNT][LAYER4_N_COUNT];
bool output_acc_First_Prop[LAYER4_N_COUNT] = { true, true, true, true, true, true, true, true, true, true };
int k = 0;


double sigmoid_func(double x);
double relu_func(double x);
double relu_der_func(double x);
double softmax_func(int index);
double softmax_func_spec(int index);
double grad_clip(double x);
void forwardPropogate(char* file_path_data, int offset_data, int label);
void backwardPropogate(int correct_label);
void spec_backwardPropogate(int correct_label);
void ApplyGradient();
void ApplySpecGradient();
void setPrevFP(int label);
int getLabel(char* file_path_labels, int offset_label);

int main(int argc, char* argv[]) {
	clock_t start, end;

	// Randomize Weights & biases of all Neurons Layers 1-3
	int WeightCount = 0;
	for (int i = 0; i < LAYER1_N_COUNT; i++) {
		for (int j = 0; j < LAYER2_N_COUNT; j++) {
			input[i].weights[j] = (double)((rand() % 21) - 10) / WEIGHT_INIT_DIVIDER;
			if (input[i].weights[j] == 0) {
				input[i].weights[j] = LEARNING_RATE;
			}
		}
	}

	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		for (int j = 0; j < LAYER3_N_COUNT; j++) {
			layer_2[i].weights[j] = (double)((rand() % 21) - 10) / WEIGHT_INIT_DIVIDER;
			if (layer_2[i].weights[j] == 0) {
				layer_2[i].weights[j] = LEARNING_RATE;
			}
		}
		layer_2[i].bias = (double)((rand() % 21) - 10) / WEIGHT_INIT_DIVIDER;
		if (layer_2[i].bias == 0) {
			layer_2[i].bias = LEARNING_RATE;
		}

		layer_2[i].biasgrad = 0;
	}

	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		for (int j = 0; j < LAYER4_N_COUNT; j++) {
			layer_3[i].weights[j] = (double)((rand() % 21) - 10) / WEIGHT_INIT_DIVIDER;
			if (layer_3[i].weights[j] == 0) {
				layer_3[i].weights[j] = LEARNING_RATE;
			}
		}
		layer_3[i].bias = (double)((rand() % 21) - 10) / WEIGHT_INIT_DIVIDER;
		if (layer_3[i].bias == 0) {
			layer_3[i].bias = LEARNING_RATE;
		}

		layer_3[i].biasgrad = 0;
	}

	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		output[i].bias = (double)((rand() % 21) - 10) / WEIGHT_INIT_DIVIDER;
		if (output[i].bias == 0) {
			output[i].bias = LEARNING_RATE;
		}

		output[i].biasgrad = 0;
	}

	/*/ Train on first image
	int current_label;
	for (int i = 0; i < 5000; i++) {
		current_label = forwardPropogate(Train_Data_File_Pixels, Train_Data_File_Labels, PIXEL_START_OFFSET, LABEL_START_OFFSET);
		backwardPropogate(current_label);
		current_label = forwardPropogate(Train_Data_File_Pixels, Train_Data_File_Labels, PIXEL_START_OFFSET + PIXELS_PER_IMAGE, LABEL_START_OFFSET + 1);
		backwardPropogate(current_label);
		ApplyGradient();
	}

	// Apply Gradient and then see the output for first image again (#5 should light up)
	current_label = forwardPropogate(Train_Data_File_Pixels, Train_Data_File_Labels, PIXEL_START_OFFSET, LABEL_START_OFFSET);
	printf("The current label is %d\n", current_label);
	current_label = forwardPropogate(Train_Data_File_Pixels, Train_Data_File_Labels, PIXEL_START_OFFSET + PIXELS_PER_IMAGE, LABEL_START_OFFSET + 1);
	printf("The current label is %d\n", current_label);*/

	// Train AI
	start = clock();
	int current_label;
	int backprop_label;
	double sum = 0;
	for (k = 0; k < EPOCH; k++) {
		for (int i = 0; i < TRAINING_IMAGES; i++) {
			current_label = getLabel(Train_Data_File_Labels, LABEL_START_OFFSET + i);
			//printf("The current GETLABEL label is %d\n", current_label);
			//printf("%d\n", i);
			// 2 Threads, 1 for forward propogation and 1 for backward propogation.
            #pragma omp parallel num_threads(2)
			{

				// 1 Thread executes forward propogation, "nowait" clause makes it continue without yielding for the end of thread.
				#pragma omp single nowait
				{
					forwardPropogate(Train_Data_File_Pixels, PIXEL_START_OFFSET + (i * PIXELS_PER_IMAGE), current_label);
					//printf("The current label is %d\n", current_label);
					//printf("%d\n", i);
				}
				if (output_acc_First_Prop[current_label] == true) {
					// If it's the first time the label has been forward propogated, yield for all other threads then execute.
					#pragma omp barrier
					#pragma omp single
					{
						backwardPropogate(current_label);
						output_acc_First_Prop[current_label] = false;
					}
				}
				else {
					// 2nd Thread executes backward propogation.
					#pragma omp single
					{
						//printf("SPEC BACK\n");
						//printf("The current label is %d\n", current_label);
						//printf("%d\n", i);
						spec_backwardPropogate(current_label);
					}

					// Wait until both previous threads are done then continue.
					#pragma omp barrier
					#pragma omp single
					{
						// Get softmax mean-diff
						sum = 0;
						//printf("Output Neurons (Label = %d, EPOCH = %d):\n", current_label, k + 1);
						for (int i = 0; i < LAYER4_N_COUNT; i++) {
							//printf("%2.4lf\n", output[i].activation);
							sum += fabs(output[i].prev_activation[current_label] - output[i].activation);
						}
						sum = sum / LAYER4_N_COUNT;

						// If below threshold, apply SpecGradient
						if (sum <= THRESHOLD) {
							ApplySpecGradient();

							// Accumulate softmax
							//printf("Output Acc Neurons (Label = %d, EPOCH = %d, i = %d):\n", current_label, k + 1, i);
							/*for (int i = 0; i < LAYER4_N_COUNT; i++) {
								output_acc[current_label][i].activation = (output[i].activation * ALPHA) + (output_acc[current_label][i].activation * BETA);
								//printf("%2.4lf\n", output_acc[current_label][i].activation);
							}*/
						}
						else {
							// Else, Use normal backprop
							backwardPropogate(current_label);
						}
					}
				}
			}

			if (i % BATCH_SIZE == 0) {
				ApplyGradient();
			}
			setPrevFP(current_label);
		}
	}
	end = clock();

	printf("Training took: %2.4lf seconds\n", (double)(end - start) / (double)CLOCKS_PER_SEC);

	// Test of Images
	int max_neuron = 0;
	int num_of_correct_images = 0;
	double max_neuron_act = 0;
	for (int i = 0; i < TEST_IMAGES; i++) {
		current_label = getLabel(Test_Data_File_Labels, LABEL_START_OFFSET + i);
		forwardPropogate(Test_Data_File_Pixels, PIXEL_START_OFFSET + (i * PIXELS_PER_IMAGE), current_label);
		max_neuron = 0;
		max_neuron_act = 0;
		for (int j = 0; j < LAYER4_N_COUNT; j++) {
			if (output[j].activation > max_neuron_act) {
				max_neuron_act = output[j].activation;
				max_neuron = j;
			}
		}

		//printf("The current label is %d\n", current_label);
		if (current_label == max_neuron) {
			//printf("Correct!\n");
			num_of_correct_images++;
		}
		else {
			//printf("Incorrect!\n");
		}
	}

	printf("The accuracy of the neural network with %d images trained and tested on %d images, is: %2.4lf%%\n", TRAINING_IMAGES, TEST_IMAGES, ((double)num_of_correct_images / (TEST_IMAGES)) * 100);

	return 0;
}

void forwardPropogate(char* file_path_data, int offset_data, int label) {

	// Get Input
	FILE* data_file;

	// Open files
	data_file = fopen(file_path_data, "rb");
	if (data_file == NULL) {
		printf("Cannot open %s\n", file_path_data);
		exit(2);
	}

	// Get Pixel data
	fseek(data_file, offset_data, SEEK_SET);
	for (int i = 0; i < LAYER1_N_COUNT; i++) {
		input[i].activation = (double)fgetc(data_file) / 255;
	}
	fclose(data_file);

	double sum = 0;

	// Activate neurons of Layer 2
	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		sum = layer_2[i].bias;
		for (int j = 0; j < LAYER1_N_COUNT; j++) {
			sum += input[j].weights[i] * input[j].activation;
		}
		layer_2[i].sum = sum;
		layer_2[i].activation = relu_func(sum);
	}


	// Activate neurons of Layer 3
	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		sum = layer_3[i].bias;
		for (int j = 0; j < LAYER2_N_COUNT; j++) {
			sum += layer_2[j].weights[i] * layer_2[j].activation;
		}
		layer_3[i].sum = sum;
		layer_3[i].activation = relu_func(sum);
	}

	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		sum = output[i].bias;
		for (int j = 0; j < LAYER3_N_COUNT; j++) {
			sum += layer_3[j].weights[i] * layer_3[j].activation;
		};
		output[i].sum = sum;
	}

	// Activate neurons of output layer
	//printf("Output neurons:\n");
	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		output[i].activation = softmax_func(i);
		//printf("%2.4lf\n", output[i].activation);
	}

	return;
};

void backwardPropogate(int correct_label) {
	double Total_softmax_denom = 0;
	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		if (i == correct_label) {
			output[i].error_der = (output[i].activation-1);
		}
		else {
			output[i].error_der = (output[i].activation);
		}
	}

	// Get gradient & Error Der L3
	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		layer_3[i].error_der = 0;
		for (int j = 0; j < LAYER4_N_COUNT; j++) {
			gradient_L3[i][j] += output[j].error_der * layer_3[i].activation;
			layer_3[i].error_der += output[j].error_der * layer_3[i].weights[j];
			//gradient_L3[i][j] = grad_clip(gradient_L3[i][j]);
		}
	}

	// Get gradient L2
	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		layer_2[i].error_der = 0;
		for (int j = 0; j < LAYER3_N_COUNT; j++) {
			gradient_L2[i][j] += layer_3[j].error_der * layer_2[i].activation * relu_der_func(layer_3[j].sum);
			layer_2[i].error_der += layer_3[j].error_der * layer_2[i].weights[j] * relu_der_func(layer_3[j].sum);
			//gradient_L2[i][j] = grad_clip(gradient_L2[i][j]);
		}
	}

	// Get gradient L1
	for (int i = 0; i < LAYER1_N_COUNT; i++) {
		for (int j = 0; j < LAYER2_N_COUNT; j++) {
			gradient_L1[i][j] += layer_2[j].error_der * input[i].activation * relu_der_func(layer_2[j].sum);
			//gradient_L1[i][j] = grad_clip(gradient_L1[i][j]);
		}
	}

	// Bias Gradients

	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		output[i].biasgrad += output[i].error_der;
		//output[i].biasgrad = grad_clip(output[i].biasgrad);
	}

	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		layer_3[i].biasgrad += layer_3[i].error_der * relu_der_func(layer_3[i].sum);
		//layer_3[i].biasgrad = grad_clip(layer_3[i].biasgrad);
	}

	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		layer_2[i].biasgrad += layer_2[i].error_der * relu_der_func(layer_2[i].sum);
		//layer_2[i].biasgrad = grad_clip(layer_2[i].biasgrad);
	}

	return;
}

double sigmoid_func(double x) {
	double result = (double)1 / ((double)1 + exp(-x));

	return result;
}

double relu_func(double x) {
	if (x > 0) {
		return x;
	}
	else {
		return x*0.01;
	}
}

double relu_der_func(double x) {
	if (x > 0) {
		return 1;
	}
	else {
		return 0.01;
	}
}

double softmax_func(int index) {
	double num = exp(output[index].sum);
	double denom = 0;

	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		denom += exp(output[i].sum);
	}
	return num / denom;
}

/*double softmax_func_spec(int index) {
	double num = exp(output[index].prev_sum);
	double denom = 0;

	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		denom += exp(output[i].prev_sum);
	}

	return num / denom;
}*/

double grad_clip(double x) {
	if (x > MAX_GRAD) {
		return MAX_GRAD;
	}
	else if (x < -MAX_GRAD) {
		return -MAX_GRAD;
	}

	return x;
}

void ApplyGradient() {
	// Apply Gradient L3
	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		for (int j = 0; j < LAYER4_N_COUNT; j++) {
			layer_3[i].weights[j] -= grad_clip(gradient_L3[i][j]) * LEARNING_RATE;
			gradient_L3[i][j] = 0;
		}
	}

	// Apply Gradient L2
	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		for (int j = 0; j < LAYER3_N_COUNT; j++) {
			layer_2[i].weights[j] -= grad_clip(gradient_L2[i][j]) * LEARNING_RATE;
			gradient_L2[i][j] = 0;
		}
	}

	// Apply Gradient L1
	for (int i = 0; i < LAYER1_N_COUNT; i++) {
		for (int j = 0; j < LAYER2_N_COUNT; j++) {
			input[i].weights[j] -= grad_clip(gradient_L1[i][j]) * LEARNING_RATE;
			gradient_L1[i][j] = 0;
		}
	}

	// Apply Bias Gradients
	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		output[i].bias -= grad_clip(output[i].biasgrad) * LEARNING_RATE;
		output[i].biasgrad = 0;
	}

	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		layer_3[i].bias -= grad_clip(layer_3[i].biasgrad) * LEARNING_RATE;
		layer_3[i].biasgrad = 0;
	}

	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		layer_2[i].bias -= grad_clip(layer_2[i].biasgrad) * LEARNING_RATE;
		layer_2[i].biasgrad = 0;
	}

	return;
}

void setPrevFP(int label) {
	for (int i = 0; i < LAYER1_N_COUNT; i++) {
		input[i].prev_activation[label] = input[i].activation;
	}

	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		layer_2[i].prev_activation[label] = layer_2[i].activation;
		layer_2[i].prev_sum[label] = layer_2[i].sum;
	}

	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		layer_3[i].prev_activation[label] = layer_3[i].activation;
		layer_3[i].prev_sum[label] = layer_3[i].sum;
	}

	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		output[i].prev_activation[label] = output[i].activation;
		output[i].prev_sum[label] = output[i].sum;
		if (output_acc_First_Prop[label] == true) {
			//output_acc[label][i].activation = output[i].activation;
		}
	}
	return;
}

void spec_backwardPropogate(int correct_label) {
	double Total_softmax_denom = 0;
	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		if (i == correct_label) {
			output[i].error_der = (output[i].prev_activation[correct_label] - 1);
		}
		else {
			output[i].error_der = (output[i].prev_activation[correct_label]);
		}
	}

	// Get gradient & Error Der L3
	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		layer_3[i].error_der = 0;
		for (int j = 0; j < LAYER4_N_COUNT; j++) {
			gradient_L3_spec[i][j] = output[j].error_der * layer_3[i].prev_activation[correct_label];
			layer_3[i].error_der += output[j].error_der * layer_3[i].weights[j];
			//gradient_L3_spec[i][j] = grad_clip(gradient_L3_spec[i][j]);
		}
	}

	// Get gradient L2
	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		layer_2[i].error_der = 0;
		for (int j = 0; j < LAYER3_N_COUNT; j++) {
			gradient_L2_spec[i][j] = layer_3[j].error_der * layer_2[i].prev_activation[correct_label] * relu_der_func(layer_3[j].prev_sum[correct_label]);
			layer_2[i].error_der += layer_3[j].error_der * layer_2[i].weights[j] * relu_der_func(layer_3[j].prev_sum[correct_label]);
			//gradient_L2_spec[i][j] = grad_clip(gradient_L2_spec[i][j]);
		}
	}

	// Get gradient L1
	for (int i = 0; i < LAYER1_N_COUNT; i++) {
		for (int j = 0; j < LAYER2_N_COUNT; j++) {
			gradient_L1_spec[i][j] = layer_2[j].error_der * input[i].prev_activation[correct_label] * relu_der_func(layer_2[j].prev_sum[correct_label]);
			//gradient_L1_spec[i][j] = grad_clip(gradient_L1_spec[i][j]);
		}
	}

	// Bias Gradients

	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		output[i].biasgrad_spec = output[i].error_der;
		//output[i].biasgrad_spec = grad_clip(output[i].biasgrad_spec);
	}

	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		layer_3[i].biasgrad_spec = layer_3[i].error_der * relu_der_func(layer_3[i].prev_sum[correct_label]);
		//layer_3[i].biasgrad_spec = grad_clip(layer_3[i].biasgrad_spec);
	}

	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		layer_2[i].biasgrad_spec = layer_2[i].error_der * relu_der_func(layer_2[i].prev_sum[correct_label]);
		//layer_2[i].biasgrad_spec = grad_clip(layer_2[i].biasgrad_spec);
	}

	return;
}

void ApplySpecGradient() {
	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		for (int j = 0; j < LAYER4_N_COUNT; j++) {
			gradient_L3[i][j] += gradient_L3_spec[i][j];
		}
	}

	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		for (int j = 0; j < LAYER3_N_COUNT; j++) {
			gradient_L2[i][j] += gradient_L2_spec[i][j];
		}
	}

	for (int i = 0; i < LAYER1_N_COUNT; i++) {
		for (int j = 0; j < LAYER2_N_COUNT; j++) {
			gradient_L1[i][j] += gradient_L1_spec[i][j];
		}
	}

	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		output[i].biasgrad += output[i].biasgrad_spec;
	}

	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		layer_3[i].biasgrad += layer_3[i].biasgrad_spec;
	}

	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		layer_2[i].biasgrad += layer_2[i].biasgrad_spec;
	}
	return;
}

int getLabel(char* file_path_labels, int offset_label) {
	FILE* label_file;

	label_file = fopen(file_path_labels, "rb");
	if (label_file == NULL) {
		printf("Cannot open %s\n", file_path_labels);
		exit(2);
	}

	// Get Label Data
	fseek(label_file, offset_label, SEEK_SET);
	int label = fgetc(label_file);
	fclose(label_file);

	return label;
}