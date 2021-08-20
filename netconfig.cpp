#include "netconfig.hpp"

// ==============================
// = Add Layer to given Network =
// ==============================
void addLayer(network_t *net, layer_t layer) {
	// Assumes that Network has been initialized with enough memory (not checked!)
	// Uses static variables -> can only be used for 1 network definition per program!
	// If layer.is_first_split_layer==true, reserves double amount of
	//     output memory to allow implicit output-channel concatenation.
	//     Use for (expand1x1) and other "output-channel split" layers.
	// If layer.is_second_split_layer==true, uses same input memory address as in
	//     last layer, and moves output address only by layer.channels_out/2
	//     to interleave with previous layer's output.
	//     Use for (expand3x3) and other "output-channel split" layers.

	// Add Layer to network
	net->layers[net->num_layers] = layer;
	net->num_layers++;
	net->total_paras += layer.paras;
	net->total_gflops += layer.gflops;
};

// =========================================
// = Print Overview Table of given Network =
// =========================================
// Print List of all Layers + Attributes + Memory Locations
#define use_KB 0
#if use_KB
#define unit "k"
#define divi 1024
#else
#define unit ""
#define divi 4
#endif

void print_layer(layer_t *layer) {
	/*
	 * layer_name, height x width x channels_in > channels_out,CONV (kernel x kernel)/stride p leakyrelu
	 *
	 *
	 * */

	printf("%6s: %3d x %-3d x %3d > %-3d, paras: %d, GFLOPS: %2.3f",
			layer->name, (int)layer->height,(int)layer->width, (int)layer->channels_in, (int)layer->channels_out
			,(int)layer->paras, (float)layer->gflops);
	printf("\n");
};

void print_layers(network_t *net) {
	for (int i = 0; i < net->num_layers; i++) {
		layer_t *layer = &net->layers[i];
		print_layer(layer);
	}
}

