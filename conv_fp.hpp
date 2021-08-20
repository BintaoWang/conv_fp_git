#include "netconfig.hpp"

using namespace std;
using namespace hls;

#define PASS 0
#define FAIL 1

#define AXI_MASTER 1
const int N_PE = 64;

//configure the single engine
#define K_MAX 3
#define S_MAX 2
#define TR_MAX 60
#define TC_MAX 60
#define TM_MAX 64
#define TN_MAX 4
#define TRI_MAX 122
#define TCI_MAX 122

//array to store the input feature of block
data_type IBRAM[TRI_MAX][TCI_MAX][TN_MAX];
data_type WBRAM[K_MAX][K_MAX][TM_MAX][TN_MAX];
data_type BBRAM[TM_MAX];
data_type OBRAM[TR_MAX][TC_MAX][TM_MAX];

void convolution_fp(
		layer_t layer,
		stream<AXI_VAL_IN> &str_in_0,
		stream<AXI_VAL_IN> &str_in_1,
		stream<AXI_VAL_IN> &str_in_2,
		stream<AXI_VAL_IN> &str_in_3,

		stream<AXI_VAL_OUT> &str_out_0,
		stream<AXI_VAL_OUT> &str_out_1,
		stream<AXI_VAL_OUT> &str_out_2,
		stream<AXI_VAL_OUT> &str_out_3,
);
