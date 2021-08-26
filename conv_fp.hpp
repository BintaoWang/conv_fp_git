#include "netconfig.hpp"

using namespace std;
using namespace hls;

#define PASS 0
#define FAIL 1

#define AXI_MASTER 1
const int N_PE = 64;

//configure the single engine
#define K_MAX 3  //kernal
#define S_MAX 2   //stride
#define TR_MAX 60  // row
#define TC_MAX 60  // col
#define TM_MAX 64  // m
#define TN_MAX 4   //m
#define TRI_MAX 122  //tri
#define TCI_MAX 122  //tci

//array to store the input feature of block
data_type IBRAM[TRI_MAX][TCI_MAX][TN_MAX];
data_type WBRAM[K_MAX][K_MAX][TM_MAX][TN_MAX];
data_type BBRAM[TM_MAX];
data_type OBRAM[TR_MAX][TC_MAX][TM_MAX];

void convolution_fp(
		layer_t layer,
		AXI_VAL_IN &str_in_0,
		AXI_VAL_IN &str_in_1,
		stream<AXI_VAL_IN> &str_in_2,
		stream<AXI_VAL_IN> &str_in_3,

		stream<AXI_VAL_OUT> &str_out_0,
		stream<AXI_VAL_OUT> &str_out_1,
		stream<AXI_VAL_OUT> &str_out_2,
		stream<AXI_VAL_OUT> &str_out_3,
);
