#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LAYER1_N_COUNT 784
#define LAYER2_N_COUNT 16
#define LAYER3_N_COUNT 16
#define LAYER4_N_COUNT 10
#define PIXELS_PER_IMAGE 784
#define TRAINING_IMAGES 60000
#define TEST_IMAGES 10000
#define LABEL_START_OFFSET 8
#define PIXEL_START_OFFSET 16
#define WEIGHT_INIT_DIVIDER 10000 // To prevents weights from being too big for the sigmoid
#define BATCH_SIZE 15
#define LEARNING_RATE 0.25

#define Train_Data_File_Pixels "mnist_data/train-images.idx3-ubyte"
#define Train_Data_File_Labels "mnist_data/train-labels.idx1-ubyte"
#define Test_Data_File_Pixels "mnist_data/t10k-images.idx3-ubyte"
#define Test_Data_File_Labels "mnist_data/t10k-labels.idx1-ubyte"

struct neuron_L1 {
	double activation;
	double weights[LAYER2_N_COUNT];
};

struct neuron_L2 {
	double activation;
	double sum;
	double weights[LAYER3_N_COUNT];
	double error_der;
};

struct neuron_L3 {
	double activation;
	double sum;
	double weights[LAYER4_N_COUNT];
	double error_der;
};

struct neuron_L4 {
	double activation;
	double sum;
	double error_der;
};

// Initialize Gradients for each layer
double gradient_L1[LAYER1_N_COUNT][LAYER2_N_COUNT] = { 0 };
double gradient_L2[LAYER2_N_COUNT][LAYER3_N_COUNT] = { 0 };
double gradient_L3[LAYER3_N_COUNT][LAYER4_N_COUNT] = { 0 };
struct neuron_L1 input[LAYER1_N_COUNT];
struct neuron_L2 layer_2[LAYER2_N_COUNT];
struct neuron_L3 layer_3[LAYER3_N_COUNT];
struct neuron_L4 output[LAYER4_N_COUNT];
double sigmoid_func(double x);
int forwardPropogate(char* file_path_data, char* file_path_labels, int offset_data, int offset_label);
void backwardPropogate(int correct_label);
void ApplyGradient();

int main(int argc, char* argv[]) {
	

	// Randomize Weights of all Neurons Layers 1-3
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
	}

	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		for (int j = 0; j < LAYER4_N_COUNT; j++) {
			layer_3[i].weights[j] = (double)((rand() % 21) - 10) / WEIGHT_INIT_DIVIDER;
			if (layer_3[i].weights[j] == 0) {
				layer_3[i].weights[j] = LEARNING_RATE;
			}
		}
	}

	/*/ Train on first image
	int current_label;
	for (int i = 0; i < 1000; i++) {
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
	int current_label;
	for (int i = 0; i < TRAINING_IMAGES; i++) {
		current_label = forwardPropogate(Train_Data_File_Pixels, Train_Data_File_Labels, PIXEL_START_OFFSET + (i * PIXELS_PER_IMAGE), LABEL_START_OFFSET + i);
		//printf("The current label is %d\n", current_label);
		backwardPropogate(current_label);
		if (i % BATCH_SIZE == 0) {
			ApplyGradient();
		}
	}

	// Test of Images
	int max_neuron = 0;
	int num_of_correct_images = 0;
	double max_neuron_act = 0;
	for (int i = 0; i < TEST_IMAGES; i++) {
		current_label = forwardPropogate(Test_Data_File_Pixels, Test_Data_File_Labels, PIXEL_START_OFFSET + (i * PIXELS_PER_IMAGE), LABEL_START_OFFSET + i);
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

	printf("The accuracy of the neural network with %d images trained and tested on %d images, is: %2.4lf%%", TRAINING_IMAGES, TEST_IMAGES, ((double)num_of_correct_images / (TEST_IMAGES))*100);


	return 0;
}

int forwardPropogate(char* file_path_data, char* file_path_labels, int offset_data, int offset_label) {
	
	// Get Input
	FILE* data_file;
	FILE* label_file;

	// Open files
	data_file = fopen(file_path_data, "rb");
	if (data_file == NULL) {
		printf("Cannot open %s\n", file_path_data);
		exit(2);
	}

	label_file = fopen(file_path_labels, "rb");
	if (label_file == NULL) {
		printf("Cannot open %s\n", file_path_labels);
		exit(2);
	}

	// Get Pixel data
	fseek(data_file, offset_data, SEEK_SET);
	for (int i = 0; i < LAYER1_N_COUNT; i++) {
		input[i].activation = (double)fgetc(data_file) / 255;
	}
	fclose(data_file);

	// Get Label Data
	fseek(label_file, offset_label, SEEK_SET);
	int label = fgetc(label_file);
	fclose(label_file);

	double sum = 0;

	// Activate neurons of Layer 2
	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		sum = 0;
		for (int j = 0; j < LAYER1_N_COUNT; j++) {
			sum += input[j].weights[i] * input[j].activation;
		}
		layer_2[i].sum = sum;
		layer_2[i].activation = sigmoid_func(sum);
	}


	// Activate neurons of Layer 3
	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		sum = 0;
		for (int j = 0; j < LAYER2_N_COUNT; j++) {
			sum += layer_2[j].weights[i] * layer_2[j].activation;
		}
		layer_3[i].sum = sum;
		layer_3[i].activation = sigmoid_func(sum);
	}

	// Activate neurons of output layer
	//printf("Output neurons:\n");
	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		sum = 0;
		for (int j = 0; j < LAYER3_N_COUNT; j++) {
			sum += layer_3[j].weights[i] * layer_3[j].activation;
		};
		output[i].activation = sigmoid_func(sum);
		output[i].sum = sum;
		//printf("%2.4lf\n", output[i].activation);
	}
	return label;
};

void backwardPropogate(int correct_label) {
	double Last_Layer_error_der = 0;
	for (int i = 0; i < LAYER4_N_COUNT; i++) {
		if (i == correct_label) {
			output[i].error_der = 2 * (output[i].activation-1);
		}
		else {
			output[i].error_der = 2 * (output[i].activation);
		}
	}

	// Get gradient & Error Der L3
	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		layer_3[i].error_der = 0;
		for (int j = 0; j < LAYER4_N_COUNT; j++) {
			gradient_L3[i][j] += output[j].error_der * layer_3[i].activation * sigmoid_func(output[j].sum) * (1 - sigmoid_func(output[j].sum));
			layer_3[i].error_der += output[j].error_der * layer_3[i].weights[j] * sigmoid_func(output[j].sum) * (1 - sigmoid_func(output[j].sum));
		}
	}

	// Get gradient L2
	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		layer_2[i].error_der = 0;
		for (int j = 0; j < LAYER3_N_COUNT; j++) {
			gradient_L2[i][j] += layer_3[j].error_der * layer_2[i].activation * sigmoid_func(layer_3[j].sum) * (1 - sigmoid_func(layer_3[j].sum));
			layer_2[i].error_der += layer_3[j].error_der * layer_2[i].weights[j] * sigmoid_func(layer_3[j].sum) * (1 - sigmoid_func(layer_3[j].sum));
		}
	}

	// Get gradient L1
	for (int i = 0; i < LAYER1_N_COUNT; i++) {
		for (int j = 0; j < LAYER2_N_COUNT; j++) {
			gradient_L1[i][j] += layer_2[j].error_der * input[i].activation * sigmoid_func(layer_2[j].sum) * (1 - sigmoid_func(layer_2[j].sum));
		}
	}

	return;
}

double sigmoid_func(double x) {
	double result = (double)1 / ((double)1 + exp(-x));

	return result;
}

void ApplyGradient() {
	// Apply Gradient L3
	for (int i = 0; i < LAYER3_N_COUNT; i++) {
		for (int j = 0; j < LAYER4_N_COUNT; j++) {
			layer_3[i].weights[j] -= gradient_L3[i][j] * LEARNING_RATE;
			gradient_L3[i][j] = 0;
		}
	}

	// Apply Gradient L2
	for (int i = 0; i < LAYER2_N_COUNT; i++) {
		for (int j = 0; j < LAYER3_N_COUNT; j++) {
			layer_2[i].weights[j] -= gradient_L2[i][j] * LEARNING_RATE;
			gradient_L2[i][j] = 0;
		}
	}

	// Apply Gradient L1
	for (int i = 0; i < LAYER1_N_COUNT; i++) {
		for (int j = 0; j < LAYER2_N_COUNT; j++) {
			input[i].weights[j] -= gradient_L1[i][j] * LEARNING_RATE;
			gradient_L1[i][j] = 0;
		}
	}

	return;
}