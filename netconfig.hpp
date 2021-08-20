#ifndef _NETCONFIG_H_
#define _NETCONFIG_H_
/*
 * @author:TofKing
 * @tell:18238465404
 * @date:2021.08.14 today is valentine's day!
 * @todo: some configure pamaters of network!
 *
 * */
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <string>

#include <cmath>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

using namespace std;

//some constant parameters of network
const int MAX_NUM_LAYERS = 120;  //max number of layers in a network
const int MAX_DIMENSION = 1000;  //max dimension of input farture map
const int MAX_BLOCK_NUM = 300;   // max number of blocks in one layer
const int MAX_CHANNELS = 1024;   //max numner of channels in one layer
const int NET_NAME_MAX_LEN = 6;         // max length of layer names
const int MEMORY_ALIGNMENT = 4 * 1024;  // align data in DRAM to 4KB borders

//dafine data type!!!
typedef float data_type;

//// =================================
//// = define a AXI_stream interface =
//// =================================
#ifndef AXI_VAL_DEF
struct AXI_VAL_IN{
	data_type data;
};
struct AXI_VAL_OUT{
	data_type data;
	bool last;
};

// ================================
// = Bit-Width Calculation MACROs =
// ================================
// NBITS(constant) = how many bits needed to represent <constant>
#define NBITS2(n) ((n & 2) ? 1 : 0)
#define NBITS4(n) ((n & (0xC)) ? (2 + NBITS2(n >> 2)) : (NBITS2(n)))
#define NBITS8(n) ((n & 0xF0) ? (4 + NBITS4(n >> 4)) : (NBITS4(n)))
#define NBITS16(n) ((n & 0xFF00) ? (8 + NBITS8(n >> 8)) : (NBITS8(n)))
#define NBITS32(n) ((n & 0xFFFF0000) ? (16 + NBITS16(n >> 16)) : (NBITS16(n)))
#define NBITS(n) ((n) == 0 ? 1 : NBITS32((n)) + 1)

//// ============================
//// = Network Type-Definitions =
//// ============================
typedef ap_uint<NBITS(MAX_DIMENSION)> dimension_t;
typedef ap_uint<NBITS(MAX_BLOCK_NUM)> number_t;
typedef ap_uint<NBITS(MAX_CHANNELS)> channel_t;

// ================
// = Struct LAYER =
// ================
// Structure that holds one single CNN layer (actually, one CONV layer)
struct layer_t{
	char name[NET_NAME_MAX_LEN+1];
	dimension_t width;
	dimension_t height;
	channel_t channels_in;
	dimension_t width_out;
	dimension_t height_out;
	channel_t channels_out;

	dimension_t Tr;
	dimension_t Tc;
	dimension_t Tm;
	dimension_t Tn; //必须是2的整数倍
	dimension_t Tri;
	dimension_t Tci;

	number_t Tr_num;
	number_t Tc_num;
	number_t Tm_num;
	number_t Tn_num;

	ap_uint<3> kernel;
	ap_uint<2> stride;
	ap_uint<2> pad;
	bool relu;
	int paras;
	float gflops;
	//initialization with none
	layer_t(): // @suppress("Class members should be properly initialized")
		width(0),
		height(0),
		channels_in(0),
		channels_out(0),
		width_out(0),
		height_out(0),

		Tr(0),
		Tc(0),
		Tm(0),
		Tn(0),
		Tri(0),
		Tci(0),
		Tr_num(0),
		Tc_num(0),
		Tn_num(0),
		Tm_num(0),
		kernel(0),
		stride(0),
		pad(0),
		relu(0)
	{
		paras =0;
		gflops = 0;
		name[0] = 0;
	};
	//initialization with parameters
	 layer_t(const char *N,
			  int w,int h,int ci,int co,
			  int tr, int tc, int tm, int tn,int tri,int tci,
			  int trn, int tcn, int tnn,int tmn,
			  int k, int s,int p,
	          bool r):
		width(w),
		height(h),
		channels_in(ci),
		channels_out(co),
		Tr(tr),
		Tc(tc),
		Tm(tm),
		Tn(tn),
		Tri(tri),
		Tci(tci),
		Tr_num(trn),
		Tc_num(tcn),
		Tn_num(tnn),
		Tm_num(tmn),
		kernel(k),
		stride(s),
		pad(p),
		relu(r)
		{
			for (int i = 0; i < NET_NAME_MAX_LEN ; i++) {
			  name[i] = N[i];
			  if (N[i] == 0) break;
			}
			name[5] = 0;
			width_out =1 + floor((float)(w +  p - k)/s);
			height_out = 1 + floor((float)(w +  p - k)/s);
			paras = ( k * k * ci * co + co ) ;
			gflops =ceil(float(2 * width_out * height_out * k * k * ci * co /1000.0 / 1000.0 /1000.0));
		};
};

// ====================
// = Struct NETWORK_T =
// ====================
// Structure that holds an entire CNN Network Defintion
struct network_t {
	layer_t *layers;
	ap_uint<7> num_layers;
	data_type *weights;
	int total_paras;
	float total_gflops;
	// default constructor: need to give max_layers and max_weights
	// allocates layers[max_layers] and weights[max_weights] on Heap
	// -> can only be used on CPU, not on FPGA
	network_t(int max_layers, int max_weights):
		num_layers(0), total_paras(0), total_gflops(0)
	{
		layers = (layer_t *)malloc((sizeof(layer_t)) * max_layers);
		weights = (data_type *)malloc((sizeof(data_type)) * max_weights);
	}
};

#endif
